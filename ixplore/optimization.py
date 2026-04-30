from __future__ import annotations

import numpy as np
from scipy.special import expit, log_expit


def pca_decompose(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project data onto its top principal components."""
    X_centered = X - X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return X_centered @ Vt[:n_components].T


def posterior_means(posteriors: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Mean reduction: (N, G) @ (G, 2) -> (N, 2)."""
    return posteriors @ X


def posterior_maps(posteriors: np.ndarray, X: np.ndarray) -> np.ndarray:
    """MAP reduction: argmax over G, index into X -> (N, 2)."""
    return X[np.argmax(posteriors, axis=1)]


def fit_logistic_newton(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    regularization: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-6,
    min_step_size: float = 1e-8,
) -> np.ndarray:
    """Fit L2-penalized logistic regression by Newton's method with backtracking.

    Each Newton step is halved until the new loss is finite and non-increasing, or
    until the step fraction falls below `min_step_size`. Iteration stops once the
    accepted step has max absolute value below `tol`. Returns `[beta_1, ..., beta_D, alpha]`.
    """
    X_aug = np.column_stack([X, np.ones(len(X))])  # (N, D+1)
    D = X_aug.shape[1]
    w = np.zeros(D)

    if sample_weight is None:
        sample_weight = np.ones(len(X))

    def neg_log_lik(w_):
        z = X_aug @ w_
        return -np.sum(sample_weight * (y * log_expit(z) + (1 - y) * log_expit(-z))) \
            + 0.5 * regularization * np.dot(w_, w_)

    loss = neg_log_lik(w)
    for _ in range(max_iter):
        p = expit(X_aug @ w)
        grad = X_aug.T @ (sample_weight * (p - y)) + regularization * w
        s = sample_weight * p * (1 - p)
        H = (X_aug.T * s) @ X_aug + regularization * np.eye(D)
        step = np.linalg.solve(H, grad)

        alpha = 1.0
        while alpha > min_step_size:
            w_new = w - alpha * step
            new_loss = neg_log_lik(w_new)
            if np.isfinite(new_loss) and new_loss <= loss:
                break
            alpha *= 0.5
        w, loss = w_new, new_loss

        if np.max(np.abs(alpha * step)) < tol:
            break

    return w.reshape(1, D)
