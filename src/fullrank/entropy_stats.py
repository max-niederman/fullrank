from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass
class ComparisonEntropyStats:
    avg_entropy: float
    max_entropy_comparison: tuple[int, int]
    max_entropy: float


def comparison_entropy_stats(samples: np.ndarray, probit_scale: float) -> ComparisonEntropyStats:
    """Compute entropy statistics for pairwise comparisons.

    Parameters
    ----------
    samples : np.ndarray
        A 2-D array of shape (d, n_samples) where *d* is the number of items and
        each column contains one posterior sample of the probit scores.

    Returns
    -------
    ComparisonEntropyStats
        A dataclass instance containing

        * ``avg_entropy`` – the mean binary entropy across all \(d\choose2) unordered
          item pairs.
        * ``max_entropy_comparison`` – the (i, j) index tuple corresponding to the
          pair with the largest entropy (with ``i < j``).
        * ``max_entropy`` – the entropy value of that most uncertain pair.
    """

    # Ensure we have a 2-D array: (d, n_samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be 2-D with shape (d, n_samples)")

    d, n = samples.shape
    if d < 2:
        # Entropy across comparisons is undefined when there is only one item.
        raise ValueError("`samples` must have at least 2 items")

    sqrt2 = np.sqrt(2.0)

    # Helper for the standard normal CDF (Ψ).
    def psi(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + special.erf(x / sqrt2))

    entropies: list[float] = []
    max_entropy = -np.inf
    max_pair: tuple[int, int] = (0, 1)

    # Iterate over all unordered pairs (i < j)
    for i in range(d - 1):
        # broadcast subtraction to obtain differences to all subsequent j indices at once
        diffs = samples[i, :] - samples[i + 1 : d, :]  # shape ((d-i-1), n)
        probs = psi(probit_scale * diffs)  # same shape

        # Monte Carlo mean over samples dimension
        p_means = probs.mean(axis=1)  # shape (d-i-1,)

        # Clip to avoid log(0)
        p_means = np.clip(p_means, 1e-12, 1.0 - 1e-12)

        # Binary entropy H(p) = -p log p - (1-p) log(1-p)
        pair_entropies = -p_means * np.log(p_means) - (1.0 - p_means) * np.log(
            1.0 - p_means
        )

        # Update lists/statistics
        entropies.extend(pair_entropies.tolist())

        # Check for max entropy in this batch
        local_max_idx = np.argmax(pair_entropies)
        local_max_entropy = pair_entropies[local_max_idx]
        if local_max_entropy > max_entropy:
            max_entropy = local_max_entropy
            max_pair = (i, int(i + 1 + local_max_idx))

    avg_entropy = float(np.mean(entropies))

    return ComparisonEntropyStats(
        avg_entropy=avg_entropy,
        max_entropy_comparison=max_pair,
        max_entropy=float(max_entropy),
    )
