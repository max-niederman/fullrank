import numpy as np
from scipy.stats import multivariate_normal, norm
import approxcdf

from fullrank import Posterior


# def mean(posterior: Posterior) -> np.ndarray:
#     """
#     Compute the mean of the posterior distribution.

#     See Eq. 30 in https://link.springer.com/article/10.1007/BF03263544
#     See Eq. 7 in https://link.springer.com/article/10.1007/s00362-021-01235-2 (with correction)
#     """
#     tau = posterior.comp_matrix @ posterior.prior_mean

#     normalization_constant = multivariate_normal.cdf(
#         tau,
#         cov=posterior.Gamma,
#     )

#     nabla_phi = norm.pdf(tau)
#     if posterior.m > 1:
#         for j in range(posterior.m):
#             tau_others = np.delete(tau, j, axis=0)
#             Gamma_others = np.delete(np.delete(posterior.Gamma, j, axis=0), j, axis=1)
#             Gamma_j_to_others = np.delete(posterior.Gamma[:, j], j)
#             tau_tilde = tau[j] * Gamma_j_to_others
#             Gamma_tilde = Gamma_others - np.outer(
#                 Gamma_j_to_others, Gamma_j_to_others
#             )
#             nabla_phi[j] *= multivariate_normal.cdf(
#                 tau_others - tau_tilde,
#                 cov=Gamma_tilde,
#             )

#     return posterior.xi + 1 / normalization_constant * posterior.Delta @ nabla_phi


def lddp(
    posterior: Posterior,
    samples: np.ndarray | int = 100,
) -> float:
    """
    Compute the LDDP of the posterior under the prior's probability measure.
    This is also the negative KL divergence from the prior to the posterior.
    """
    if isinstance(samples, int):
        samples = posterior.sample(samples)
    else:
        samples = samples

    # τ = D μ
    tau = posterior.comp_matrix @ posterior.prior_mean

    # ln Z = ln Phi(τ; Γ)
    log_normalization_constant = multivariate_normal.logcdf(
        tau,
        cov=posterior.Gamma,
    )

    # E[ln Phi(D x)]
    log_likelihood_ratio = sum(
        approxcdf.mvn_cdf(
            posterior.comp_matrix @ sample,
            np.eye(posterior.m),
            is_standardized=True,
            logp=True,
        )
        for sample in samples.T
    ) / len(samples.T)

    # -E[ln p(x|D) / p(x)] = -E[ln Phi(D x) / Z]
    return -(log_likelihood_ratio - log_normalization_constant)


def comparison_skewness_norms(posterior: Posterior) -> np.ndarray:
    """
    Compute the L2 norms of the skewness vectors of each comparison in the posterior.
    Returns a 2D array of shape (n, n) where the (i, j) entry is the L2 norm of the skewness vector of the comparison between items i and j.
    """
    n = posterior.prior_mean.shape[0]

    all_comparisons = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            all_comparisons[i, j, i] += 1.0
            all_comparisons[i, j, j] -= 1.0

    Delta_squared = posterior.Delta @ posterior.Delta.T

    return np.einsum(
        "ijk,kl,ijl->ij",
        all_comparisons,
        Delta_squared,
        all_comparisons,
    )
