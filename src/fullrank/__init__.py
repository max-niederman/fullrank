import numpy as np
from fullrank.comparison import Comparison
from fullrank.truncated_mvn import TruncatedMVN


def infer(
    prior_mean: np.ndarray,
    prior_cov: np.ndarray,
    comparisons: list[Comparison],
    probit_scale: float = 1.0,
) -> "Posterior":
    comp_matrix = np.zeros((len(comparisons), prior_mean.shape[0]))
    for i, comparison in enumerate(comparisons):
        comp_matrix[i, comparison.winner] = 1
        comp_matrix[i, comparison.loser] = -1
    return Posterior(prior_mean, prior_cov, comp_matrix, probit_scale)


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
        probit_scale: float,
    ):
        m = comp_matrix.shape[0]

        self.prior_mean = prior_mean
        self.prior_cov = prior_cov
        self.comp_matrix = comp_matrix
        self.probit_scale = probit_scale
        self.xi = prior_mean  # ξ <- μ
        self.Delta = prior_cov.transpose() @ comp_matrix.transpose()  # Δ <- ΣᵀD
        self.Gamma = (
            np.eye(m) / probit_scale**2
            + comp_matrix @ np.linalg.inv(prior_cov) @ comp_matrix.transpose()
        )  # Γ <- I/β² + DΣ⁻¹Dᵀ
        self.Delta_times_Gamma_inv = self.Delta @ np.linalg.inv(self.Gamma)  # ΔΓ⁻¹
        self.Psi_bar = (
            prior_cov - self.Delta_times_Gamma_inv @ self.Delta.transpose()
        )  # Ψ̄ <- Σ - ΔΓ⁻¹Δᵀ
        self.U1_dist = TruncatedMVN(
            np.zeros(m),
            self.Gamma,
            -comp_matrix @ prior_mean,
            np.inf * np.ones(m),
        )

    def sample(self, num_samples: int) -> np.ndarray:
        xi = np.tile(self.xi, (num_samples, 1)).transpose()
        U0 = np.random.multivariate_normal(
            np.zeros(self.xi.shape[0]), self.Psi_bar, num_samples
        ).transpose()
        U1 = self.U1_dist.sample(num_samples)
        return xi + U0 + self.Delta_times_Gamma_inv @ U1
