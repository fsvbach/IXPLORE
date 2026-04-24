from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal


def set_gaussian_prior(
    X: np.ndarray,
    prior_variance: float = 1.0,
    prior_mean: np.ndarray = np.array([0, 0]),
    log: bool = False,
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
    log : bool
        If True, return the log of the prior. Default is False.

    Returns
    -------
    np.ndarray
        The (log) normalized prior distribution of shape (G,).
    """
    prior_cov = np.eye(2) * prior_variance
    prior = multivariate_normal(prior_mean, prior_cov)
    if log:
        return prior.logpdf(X)
    prior_X = prior.pdf(X)
    return prior_X / prior_X.sum()


def set_log_barrier_prior(
    X: np.ndarray,
    alpha: float = 1.0,
    limits: tuple[float, float] = (-1.0, 1.0),
    log: bool = False,
) -> np.ndarray:
    """Create a log-barrier prior over the square 2D grid.

    The penalty is R(x, y) = -alpha * [log(1 - u^2) + log(1 - v^2)], where
    u and v are the grid coordinates rescaled to (-1, 1) using the given
    limits. The prior diverges at the boundary and is flat near the centre.

    Parameters
    ----------
    X : np.ndarray
        The grid points of shape (G, 2).
    alpha : float
        Barrier strength. Larger values push mass away from the boundary.
    limits : tuple[float, float]
        (min, max) limits applied to both axes, used to rescale X to (-1, 1). Default (-1, 1).
    log : bool
        If True, return the (unnormalized) log-prior. Default False.

    Returns
    -------
    np.ndarray
        The (log) prior distribution of shape (G,). In non-log form it is
        normalized to sum to 1; in log form it is the shifted log-prior.
    """
    lo, hi = limits
    u = 2 * (X[:, 0] - lo) / (hi - lo) - 1
    v = 2 * (X[:, 1] - lo) / (hi - lo) - 1
    interior = (np.abs(u) < 1.0) & (np.abs(v) < 1.0)
    log_prior = np.full(X.shape[0], -np.inf)
    log_prior[interior] = alpha * (
        np.log(1 - u[interior] ** 2) + np.log(1 - v[interior] ** 2)
    )
    if log:
        return log_prior
    shifted = log_prior - log_prior[interior].max()
    prior_X = np.exp(shifted)
    prior_X[~interior] = 0.0
    return prior_X / prior_X.sum()
