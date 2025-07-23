from dataclasses import dataclass
import numpy as np
from textual.app import App, ComposeResult
from textual.containers import HorizontalGroup, Center
from textual.widgets import Header, Footer, Button, ProgressBar, Static

from fullrank import infer, posterior_stats, Comparison


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
        prior_var: float = 1.0,
    ):
        super().__init__()
        self.items = items
        self.comparisons: list[Comparison] = []
        self.left_index = 0
        self.right_index = 1
        self.prior_var = prior_var

    def compose(self) -> ComposeResult:
        yield Header(name="Fullrank", show_clock=True)
        with Center(id="entropy-stats"):
            yield Static("KL(Posterior || Prior)")
            yield ProgressBar(
                total=1.0, show_eta=False, show_percentage=False, id="entropy-bar"
            )
            yield Static("0.00", id="entropy-value")
            yield Static(" bits")
        with HorizontalGroup(id="comparison-buttons"):
            yield Button(self.items[self.left_index], action="left", id="left-button")
            yield Button(
                self.items[self.right_index], action="right", id="right-button"
            )
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
        posterior = infer(
            np.zeros(len(self.items)),
            self.prior_var * np.eye(len(self.items)),
            self.comparisons,
        )

        comp_skewness_norms = posterior_stats.comparison_skewness_norms(posterior)
        np.fill_diagonal(comp_skewness_norms, np.inf)
        self.left_index, self.right_index = map(
            int,
            np.unravel_index(np.argmin(comp_skewness_norms), comp_skewness_norms.shape),
        )
        self.query_one("#left-button").label = self.items[self.left_index]
        self.query_one("#right-button").label = self.items[self.right_index]

        kl_div = -posterior_stats.lddp(posterior, samples=100)
        self.query_one("#entropy-value").update(f"{kl_div:.2f}")
        self.query_one("#entropy-bar").update(progress=1.0 - np.exp(-kl_div))

    def action_quit(self) -> None:
        self.exit(self.comparisons)
