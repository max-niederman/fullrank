from pathlib import Path
import sys
import numpy as np
import typer
from rich import print

import fullrank
from fullrank.comparison_tui import ComparisonApp

app = typer.Typer()


@app.command()
def infer(
    items_file: Path = typer.Argument(
        ..., help="File containing the items to rank, seperated by newlines"
    ),
    probit_scale: float = typer.Option(
        3.0, help="The scale parameter for the probit function"
    ),
):
    items = [line.rstrip("\n") for line in items_file.read_text().splitlines()]

    comparisons = ComparisonApp(items, probit_scale=probit_scale).run()
    if comparisons is None:
        print("[bold red]No comparisons were made.[/bold red]")
        return

    print(comparisons)

    posterior = fullrank.infer(
        np.zeros(len(items)), np.eye(len(items)), comparisons, probit_scale=probit_scale
    )


@app.command()
def sample(): ...


if __name__ == "__main__":
    app()
