import numpy as np
from typing import TypedDict
from fullrank.truncated_mvn import TruncatedMVN


class Comparison(TypedDict):
    winner: int
    loser: int

def infer(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    comparisons: list[Comparison],
) -> "Posterior":
    comp_matrix = np.zeros((len(comparisons), prior_mean.shape[0]))
    for i, comparison in enumerate(comparisons):
        comp_matrix[i, comparison["winner"]] = 1
        comp_matrix[i, comparison["loser"]] = -1
    return Posterior(prior_mean, prior_cov, comp_matrix)


class Posterior:
    """
    A posterior distribution over the scores of the items under the Thurstonian model.
    Takes the form of a unified skew normal distribution.

    Reference https://link.springer.com/article/10.1007/s00362-021-01235-2 for SUN notational conventions and background on the sampling procedure.
    """

    def __init__(
        self,
        prior_mean: np.ndarray,
        prior_cov: np.ndarray,
        comp_matrix: np.ndarray,
    ):
        self.m = comp_matrix.shape[0]
        self.n = comp_matrix.shape[1]

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.comp_matrix = comp_matrix
        self.xi = prior_mean  # ξ <- μ
        self.Delta = prior_cov @ comp_matrix.T  # Δ <- ΣDᵀ
        self.Gamma = (
            np.eye(self.m) + comp_matrix @ prior_cov @ comp_matrix.T
        )  # Γ <- I + DΣDᵀ
        self.Delta_times_Gamma_inv = self.Delta @ np.linalg.inv(self.Gamma)  # ΔΓ⁻¹
        self.Psi_bar = (
            prior_cov - self.Delta_times_Gamma_inv @ self.Delta.T
        )  # Ψ̄ <- Σ - ΔΓ⁻¹Δᵀ
        self.U1_dist = TruncatedMVN(
            np.zeros(self.m),
            self.Gamma,
            -comp_matrix @ prior_mean,
            np.inf * np.ones(self.m),
        )

    def sample(self, num_samples: int) -> np.ndarray:
        """
        Samples columns from the posterior distribution.
        Returns a matrix of shape (n, num_samples).
        """
        xi = np.tile(self.xi, (num_samples, 1)).transpose()
        U0 = np.random.multivariate_normal(
            np.zeros(self.xi.shape[0]), self.Psi_bar, num_samples
        ).T
        U1 = self.U1_dist.sample(num_samples)
        return xi + U0 + self.Delta_times_Gamma_inv @ U1
