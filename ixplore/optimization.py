from __future__ import annotations

import numpy as np
from scipy.special import expit


def pca_decompose(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project data onto its top principal components."""
    X_centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    regularization: float = 1e-6,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit logistic regression via Newton's method with soft labels and sample weights.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Feature matrix (may be raw 2D coordinates or transformed features).
    y : np.ndarray of shape (N,)
        Labels in [0, 1] (soft labels allowed).
    sample_weight : np.ndarray of shape (N,) or None
        Per-sample weights. If None, uniform weights are used.
    regularization : float
        L2 regularization strength (small = weak regularization).
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    np.ndarray of shape (1, D+1)
        Fitted parameters [beta1, ..., betaD, alpha].
    """
    X_aug = np.column_stack([X, np.ones(len(X))])  # (N, D+1)
    D = X_aug.shape[1]
    w = np.zeros(D)

    if sample_weight is None:
        sample_weight = np.ones(len(X))

    for _ in range(max_iter):
        p = expit(X_aug @ w)
        grad = X_aug.T @ (sample_weight * (p - y)) + regularization * w
        s = sample_weight * p * (1 - p)
        H = (X_aug.T * s) @ X_aug + regularization * np.eye(D)
        w -= np.linalg.solve(H, grad)

    return w.reshape(1, D)
