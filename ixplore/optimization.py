from __future__ import annotations

import numpy as np
from scipy.special import expit


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    C: float = 1e6,
    max_iter: int = 50,
) -> np.ndarray:
    """Fit logistic regression via Newton's method with soft labels and sample weights.

    Parameters
    ----------
    X : np.ndarray of shape (N, 2)
        Feature matrix (grid coordinates).
    y : np.ndarray of shape (N,)
        Labels in [0, 1] (soft labels allowed).
    sample_weight : np.ndarray of shape (N,) or None
        Per-sample weights. If None, uniform weights are used.
    C : float
        Inverse regularization strength (large = weak regularization).
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    np.ndarray of shape (1, 3)
        Fitted parameters [beta1, beta2, alpha].
    """
    X_aug = np.column_stack([X, np.ones(len(X))])  # (N, 3)
    w = np.zeros(3)

    if sample_weight is None:
        sample_weight = np.ones(len(X))

    for _ in range(max_iter):
        p = expit(X_aug @ w)
        grad = X_aug.T @ (sample_weight * (p - y)) + w / C
        s = sample_weight * p * (1 - p)
        H = (X_aug.T * s) @ X_aug + np.eye(3) / C
        w -= np.linalg.solve(H, grad)

    return w.reshape(1, 3)
