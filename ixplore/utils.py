from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm


def add_sparsity(
    df: pd.DataFrame,
    keep_fraction: float | np.ndarray | list,
    generator: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """Randomly set entries to NaN to simulate missing data.

    Parameters
    ----------
    keep_fraction: float or array-like of shape (n_rows,)
        Fraction of entries to keep in [0, 1]. If array-like, sets the keep fraction per row.
    generator: np.random.Generator
        Random generator for reproducibility.
    """
    values = df.values
    fractions = np.broadcast_to(np.asarray(keep_fraction).reshape(-1, 1), values.shape)
    mask = generator.random(values.shape) < fractions
    sparse_array = np.where(mask, values, np.nan)
    return pd.DataFrame(sparse_array, index=df.index, columns=df.columns)


sparsen = add_sparsity  # backward compatibility


def add_noise(
    df: pd.DataFrame,
    keep_fraction: float | np.ndarray | list,
    generator: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """Randomly flip entries to a different valid value, inferred per column from unique values.

    Parameters
    ----------
    keep_fraction: float or array-like of shape (n_rows,)
        Fraction of entries to keep unchanged in [0, 1]. If array-like, sets the keep fraction per row.
    generator: np.random.Generator
        Random generator for reproducibility.
    """
    values = df.values
    fractions = np.broadcast_to(np.asarray(keep_fraction).reshape(-1, 1), values.shape)
    mask = generator.random(values.shape) < fractions
    # For each column, sample a random *different* value from that column's uniques
    replacements = np.empty_like(values)
    for c in range(values.shape[1]):
        unique = np.unique(values[:, c])
        n = len(unique)
        # Sample indices in [0, n-1), then shift up past the current value's index
        idx = generator.integers(0, n - 1, size=values.shape[0]) if n > 1 else np.zeros(values.shape[0], dtype=int)
        current_idx = np.searchsorted(unique, values[:, c])
        replacements[:, c] = unique[(current_idx + 1 + idx) % n]
    noisy_array = np.where(mask, values, replacements)
    return pd.DataFrame(noisy_array, index=df.index, columns=df.columns)


def scale_reactions(
    reactions: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0,
    ) -> np.ndarray:
    """Scale reactions linearly into [min_value, max_value]."""
    scaled = (reactions - np.nanmin(reactions)) / (np.nanmax(reactions) - np.nanmin(reactions))
    return scaled * (max_value - min_value) + min_value

def binarize_reactions(
    array: np.ndarray,
    generator: np.random.Generator = np.random.default_rng(0),
) -> np.ndarray:
    """Binarize an array by Bernoulli sampling with per-entry probabilities."""
    random_matrix = generator.random(array.shape)
    return (random_matrix <= array).astype(int)


def add_ones(array: np.ndarray) -> np.ndarray:
    """Append a column of ones to `array` for an intercept term."""
    return np.hstack((array, np.ones((len(array),1))))


def create_meshgrid(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    sampling_resolution: int,
) -> np.ndarray:
    """Return a (G, 2) meshgrid where G = sampling_resolution**2.

    Parameters
    ----------
    xlim: tuple
        (min, max) limits along the x-axis.
    ylim: tuple
        (min, max) limits along the y-axis.
    sampling_resolution: int
        Number of points per axis.
    """
    x_axis = np.linspace(xlim[0], xlim[1], sampling_resolution)
    y_axis = np.linspace(ylim[0], ylim[1], sampling_resolution)
    xx, yy = np.meshgrid(x_axis, y_axis)
    return np.c_[xx.ravel(), yy.ravel()]


def transformation_matrix(
    rotate: float = 0.0,
    scale: tuple[float, float] = (1.0, 1.0),
    shear: float = 0.0,
    order: tuple[Literal["shear", "rotate", "scale"], ...] = ("shear", "rotate", "scale"),
) -> np.ndarray:
    """Build a 2x2 linear transformation from rotation, scaling, and shear.

    Parameters
    ----------
    rotate: float
        Rotation angle in degrees (counterclockwise).
    scale: tuple
        Scaling factors (sx, sy).
    shear: float
        Shear factor along the x-axis.
    order: tuple
        Order in which 'scale', 'shear', and 'rotate' are composed.
    """
    R = np.array([[np.cos(np.radians(rotate)), -np.sin(np.radians(rotate))],
                    [np.sin(np.radians(rotate)),  np.cos(np.radians(rotate))]])
    S = np.diag(scale)
    Sh = np.array([[1, shear],
                    [0, 1]])
    mats = {"rotate": R, "scale": S, "shear": Sh}
    M = np.eye(2)
    for step in order:
        M = mats[step] @ M

    return M


def compute_rasch_values(
    scores: np.ndarray,
    num_options: int = 5,
    variance: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Rasch probabilities over `num_options` answer options for each score.

    Parameters
    ----------
    num_options: int
        Number of evenly spaced answer options in [0, 1].
    variance: float
        Variance of the normal kernel used to build the Rasch probabilities.

    Returns
    -------
    tuple
        (probabilities of shape (num_options, len(scores)), answer_options of shape (num_options,)).
    """
    answer_options = np.linspace(0,1,num_options)
    if num_options == 2:
        return np.array([1 - scores, scores]), answer_options
    else:
        values = np.array([norm.pdf(scores, mu, variance) for mu in answer_options])
        return values / values.sum(axis=0), answer_options
    
def mean_impute(
    X: np.ndarray,
    column_means: np.ndarray | None = None,
    fill_empty_columns: float | None = None,
) -> np.ndarray:
    """Replace NaN values with column means. If `column_means` is given, those are used directly; otherwise computed from X. Raises on all-NaN columns unless fill_empty_columns is given."""
    if column_means is None:
        all_nan_cols = np.where(np.all(np.isnan(X), axis=0))[0]
        if all_nan_cols.size and fill_empty_columns is None:
            raise ValueError(f"mean_impute: columns {all_nan_cols.tolist()} are entirely NaN")
        column_means = np.nanmean(X, axis=0)
        if all_nan_cols.size:
            column_means[all_nan_cols] = fill_empty_columns
    X_imputed = X.copy()
    nan_mask = np.isnan(X_imputed)
    X_imputed[nan_mask] = np.take(column_means, np.where(nan_mask)[1])
    return X_imputed


def iterative_pca_impute(
    X: np.ndarray,
    n_components: int = 2,
    max_iter: int = 5,
    fill_empty_columns: float | None = None,
) -> np.ndarray:
    """Replace NaN values with iterative PCA reconstruction (Grung & Manne 1998). Raises on all-NaN columns unless fill_empty_columns is given."""
    mask = np.isnan(X)
    if not mask.any():
        return X.copy()

    all_nan_cols = np.where(np.all(mask, axis=0))[0]
    if all_nan_cols.size and fill_empty_columns is None:
        raise ValueError(f"iterative_pca_impute: columns {all_nan_cols.tolist()} are entirely NaN")

    col_means = np.nanmean(X, axis=0)
    if all_nan_cols.size:
        col_means[all_nan_cols] = fill_empty_columns
    X_filled = X.copy()
    X_filled[mask] = np.take(col_means, np.where(mask)[1])

    for _ in range(max_iter):
        X_centered = X_filled - X_filled.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        scores = X_centered @ Vt[:n_components].T
        recon = scores @ Vt[:n_components] + X_filled.mean(axis=0)
        X_filled[mask] = recon[mask]

    return X_filled


def normalize_embedding(
    embedding: np.ndarray,
    limits: tuple[float, float] = (-1.0, 1.0),
    scaling: float = 1.05,
) -> np.ndarray:
    """Center and scale the embedding to fit within the given square limits.

    The embedding is centered on the limit midpoint and scaled so its extreme
    point sits at ``half_range / scaling`` from the center, leaving a margin
    of ``(scaling - 1)`` inside the limits.
    """
    if embedding is None:
        raise ValueError("Embedding must be initialized before normalizing.")
    limit_center = (limits[0] + limits[1]) / 2
    limit_half_range = (limits[1] - limits[0]) / 2
    centroid = (embedding.max(axis=0) + embedding.min(axis=0)) / 2
    embedding -= centroid
    max_extent = np.abs(embedding).max(axis=0) * scaling
    return embedding / max_extent * limit_half_range + limit_center