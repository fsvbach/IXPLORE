from __future__ import annotations

import numpy as np


def posterior_means(posteriors: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Mean reduction: (N, G) @ (G, 2) -> (N, 2)."""
    return posteriors @ X


def posterior_maps(posteriors: np.ndarray, X: np.ndarray) -> np.ndarray:
    """MAP reduction: argmax over G, index into X -> (N, 2)."""
    return X[np.argmax(posteriors, axis=1)]
