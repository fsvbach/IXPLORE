"""Likelihood functions for posterior computation."""

from __future__ import annotations

import numpy as np


def l1_log_likelihood(
    answers: np.ndarray,
    likelihood: np.ndarray,
    eps: float = 1e-300,
) -> np.ndarray:
    """Compute log-likelihood using the L1-norm distance measure.

    L(x) = prod_k [1 - |a_k - p_k(x)|]

    Parameters
    ----------
    answers : np.ndarray
        Answer values of shape (K,).
    likelihood : np.ndarray
        Predicted probabilities on the grid of shape (G, K).
    eps : float
        Clipping bound to avoid log(0).

    Returns
    -------
    np.ndarray
        Log-likelihood of shape (G,).
    """
    return np.sum(np.log(1 - np.abs(answers - likelihood) + eps), axis=1)


def bce_log_likelihood(
    answers: np.ndarray,
    likelihood: np.ndarray,
    eps: float = 1e-15,
) -> np.ndarray:
    """Compute log-likelihood using the Bernoulli cross-entropy.

    L(x) = prod_k p_k(x)^a_k * (1 - p_k(x))^(1 - a_k)

    Parameters
    ----------
    answers : np.ndarray
        Answer values of shape (K,).
    likelihood : np.ndarray
        Predicted probabilities on the grid of shape (G, K).
    eps : float
        Clipping bound to avoid log(0).

    Returns
    -------
    np.ndarray
        Log-likelihood of shape (G,).
    """
    p = np.clip(likelihood, eps, 1 - eps)
    return np.sum(answers * np.log(p) + (1 - answers) * np.log(1 - p), axis=1)
