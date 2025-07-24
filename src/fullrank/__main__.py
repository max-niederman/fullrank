import json
from pathlib import Path
import sys
import numpy as np
import typer
from rich import print
from rich.progress import track

import fullrank
from fullrank import posterior_stats
from fullrank.comparison_tui import ComparisonApp

app = typer.Typer()


@app.command()
def compare(
    items_file: Path = typer.Argument(
        ..., help="File containing the items to rank, seperated by newlines"
    ),
    prior_var: float = typer.Option(1.0, help="The variance of the prior"),
):
    """
    Compare items and write the comparisons to stdout in JSON format for inference.
    """

    items = [line.rstrip("\n") for line in items_file.read_text().splitlines()]

    comparisons = ComparisonApp(items, prior_var=prior_var).run()
    if comparisons is None:
        print("[bold red]No comparisons were made.[/bold red]", file=sys.stderr)
        return

    print(
        f"[bold green]Finished {len(comparisons)} comparisons of {len(items)} items.[/bold green]",
        file=sys.stderr,
    )

    print(
        json.dumps(
            {"items": items, "prior_var": prior_var, "comparisons": comparisons},
            indent="\t",
        )
    )


@app.command()
def raw_sample(
    n: int = typer.Argument(100_000, help="The number of samples to draw"),
    output_file: Path = typer.Argument(..., help="The file to write the samples to"),
    batch_size: int = typer.Option(
        1000, help="The number of samples to draw per batch"
    ),
):
    """
    Sample from the posterior distribution and write to a file in JSONL format.
    """
    compare_result = json.loads(sys.stdin.read())
    posterior = fullrank.infer(
        np.zeros(len(compare_result["items"])),
        compare_result["prior_var"] * np.eye(len(compare_result["items"])),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    print(f"[bold]Output File:[/bold]", output_file, file=sys.stderr)

    with open(output_file, "w") as f:
        for _ in track(range(0, n, batch_size), description="[blue]Sampling...[/blue]"):
            for sample in posterior.sample(batch_size).T:
                f.write(json.dumps(sample.tolist()) + "\n")


@app.command()
def stats(
    n: int = typer.Argument(100_000, help="The number of samples to draw"),
    entropy: bool = typer.Option(False, flag_value=True, help="Compute entropy"),
):
    """
    Compute statistics from a posterior distribution.
    """

    compare_result = json.loads(sys.stdin.read())
    items = compare_result["items"]

    posterior = fullrank.infer(
        np.zeros(len(items)),
        compare_result["prior_var"] * np.eye(len(items)),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    batch_size = 1000
    batches = []
    for _ in track(range(0, n, batch_size), description="[blue]Sampling...[/blue]"):
        batches.append(posterior.sample(batch_size))
    samples = np.concatenate(batches, axis=1)
    del batches

    print("[bold]Mean:[/bold] ", samples.mean(axis=1))

    sorted_indices = np.argsort(samples, axis=0)
    sorted_indices.sort(axis=1)
    print(
        "[bold]Ranking Deciles:[/bold]",
        sorted_indices[:, np.arange(0, n, n // 10)],
        sep="\n",
    )
    print(
        "[bold]Ranking Probabilities (rows are items, columns are rankings):[/bold]",
        np.stack([np.bincount(row, minlength=len(items)) for row in sorted_indices])
        / n,
        sep="\n",
    )

    if entropy:
        print(
            "[bold]Entropy:[/bold] ",
            posterior_stats.lddp(posterior, samples=samples),
        )


if __name__ == "__main__":
    app()
