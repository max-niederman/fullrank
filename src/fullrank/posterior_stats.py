from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.stats import multivariate_normal

from fullrank import Posterior


def lddp(posterior: Posterior, samples: np.ndarray | int = 100) -> float:
    """
    Compute the KL divergence of the posterior from the prior.
    This is also the LDDP of the posterior under the prior's probability measure.
    """
    if isinstance(samples, int):
        samples = posterior.sample(samples)
    else:
        samples = samples

    D = posterior.probit_scale * posterior.comp_matrix
    m = D.shape[0]

    # ln Z = ln Phi(D mu, I + D Sigma D^T)
    log_normalization_constant = multivariate_normal.logcdf(
        (D @ posterior.prior_mean).T,
        cov=np.eye(posterior.comp_matrix.shape[0]) + D @ posterior.prior_cov @ D.T,
    )

    # E[ln Phi(D x)/phi(D x)]
    log_likelihood_ratio = multivariate_normal.logpdf(
        (D @ samples).T, cov=np.eye(m)
    ).mean()

    return -log_likelihood_ratio + log_normalization_constant


def comparison_skewness_norms(posterior: Posterior) -> np.ndarray:
    """
    Compute the L2 norms of the skewness vectors of each comparison in the posterior.
    Returns a 2D array of shape (n, n) where the (i, j) entry is the L2 norm of the skewness vector of the comparison between items i and j.
    """
    n = posterior.prior_mean.shape[0]

    all_comparisons = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i):
            all_comparisons[i, j, i] = 1.0
            all_comparisons[i, j, j] = -1.0
            all_comparisons[j, i, j] = 1.0
            all_comparisons[j, i, i] = -1.0

    Delta_squared = posterior.Delta @ posterior.Delta.T

    return np.einsum(
        "ijk,kl,ijl->ij",
        all_comparisons,
        Delta_squared,
        all_comparisons,
    )
