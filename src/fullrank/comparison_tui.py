from dataclasses import dataclass
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, Center
from textual.widgets import Header, Footer, Button, ProgressBar, Static

from fullrank import infer
from fullrank.comparison import Comparison
from fullrank.entropy_stats import comparison_entropy_stats


class ComparisonApp(App[list[Comparison]]):
    BINDINGS = [
        ("f", "left", "Select the left item"),
        ("j", "right", "Select the right item"),
        ("z", "undo", "Undo the last comparison"),
        ("q", "quit", "Finish comparing items"),
    ]
    CSS = """
        #entropy-stats {
            layout: horizontal;
        }

        ProgressBar {
            margin: 0 2;
        }

        Static {
            width: auto;
        }

        #comparison-buttons {
            width: 100%;
            height: 1fr;
        }

        Button {
            width: 50%;
            height: 100%;
        }
    """

    def __init__(
        self,
        items: list[str],
        probit_scale: float = 1.0,
    ):
        super().__init__()
        self.items = items
        self.comparisons: list[Comparison] = []
        self.left_index = 0
        self.right_index = 1
        self.probit_scale = probit_scale

    def compose(self) -> ComposeResult:
        yield Header(name="Fullrank", show_clock=True)
        with Center(id="entropy-stats"):
            yield Static("Average Comparison Entropy")
            yield ProgressBar(total=1.0, show_eta=False, show_percentage=False, id="entropy-bar")
            yield Static("1.00", id="entropy-value")
            yield Static(" bits")
        with HorizontalGroup(id="comparison-buttons"):
            yield Button(self.items[self.left_index], action="left", id="left-button")
            yield Button(self.items[self.right_index], action="right", id="right-button")
        yield Footer()

    def action_left(self) -> None:
        self.comparisons.append(
            Comparison(winner=self.left_index, loser=self.right_index)
        )
        self.next_comparison()

    def action_right(self) -> None:
        self.comparisons.append(
            Comparison(winner=self.right_index, loser=self.left_index)
        )
        self.next_comparison()

    def action_undo(self) -> None:
        self.comparisons.pop()
        self.next_comparison()

    def next_comparison(self) -> None:
        posterior = infer(np.zeros(len(self.items)), np.eye(len(self.items)), self.comparisons, probit_scale=self.probit_scale)
        entropy_stats = comparison_entropy_stats(posterior.sample(100), posterior.probit_scale)

        avg_entropy_bits = entropy_stats.avg_entropy / np.log(2)
        self.query_one("#entropy-value").update(f"{avg_entropy_bits:.2f}")
        self.query_one("#entropy-bar").update(progress=avg_entropy_bits)

        self.left_index, self.right_index = entropy_stats.max_entropy_comparison
        self.query_one("#left-button").label = self.items[self.left_index]
        self.query_one("#right-button").label = self.items[self.right_index]
    
    def action_quit(self) -> None:
        self.exit(self.comparisons)