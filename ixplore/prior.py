from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def set_gaussian_prior(
    X: np.ndarray,
    prior_variance: float = 1.0,
    prior_mean: np.ndarray = np.array([0, 0]),
) -> np.ndarray:
    """Create a Gaussian prior over the 2D grid.

    Parameters
    ----------
    X : np.ndarray
        The grid points of shape (G, 2).
    prior_variance : float
        The diagonal entry of the covariance matrix. Default is 1.0.
    prior_mean : np.ndarray
        The mean of the prior distribution. Default is [0, 0].

    Returns
    -------
    np.ndarray
        The normalized prior distribution of shape (G,).
    """
    prior_cov = np.eye(2) * prior_variance
    prior = multivariate_normal(prior_mean, prior_cov)
    prior_X = prior.pdf(X)
    return prior_X / prior_X.sum()
