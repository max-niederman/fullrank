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
    items = [line.rstrip("\n") for line in items_file.read_text().splitlines()]

    comparisons = ComparisonApp(items, prior_var=prior_var).run()
    if comparisons is None:
        print("[bold red]No comparisons were made.[/bold red]", file=sys.stderr)
        return

    print(f"[bold green]Finished {len(comparisons)} comparisons of {len(items)} items.[/bold green]", file=sys.stderr)

    print(json.dumps({"items": items, "prior_var": prior_var, "comparisons": comparisons}, indent="\t"))


@app.command()
def sample(n: int = typer.Argument(100_000, help="The number of samples to draw")):
    compare_result = json.loads(sys.stdin.read())
    posterior = fullrank.infer(
        np.zeros(len(compare_result["items"])),
        compare_result["prior_var"] * np.eye(len(compare_result["items"])),
        compare_result["comparisons"],
    )
    print("[bold green]Finished inferring posterior.[/bold green]", file=sys.stderr)

    batch_size = 1000
    batches = []
    for _ in track(range(0, n, batch_size), description="[blue]Sampling...[/blue]"):
        batches.append(posterior.sample(batch_size))
    samples = np.concatenate(batches, axis=1)

    print("[bold]Mean:[/bold] ", samples.mean(axis=1))
    print("[bold]Entropy:[/bold] ", posterior_stats.lddp(posterior, samples=samples))


if __name__ == "__main__":
    app()
