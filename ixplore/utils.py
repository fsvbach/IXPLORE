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
    """
    Randomly set a fraction of the entries in the DataFrame to NaN, simulating missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sparsified.
    keep_fraction : float or array-like of shape (n_rows,)
        The fraction of entries to keep (between 0 and 1).
        If array-like, each element sets the keep fraction for the corresponding row.
    generator : np.random.Generator, optional
        A random number generator for reproducibility.
        Default is np.random.default_rng().

    Returns
    -------
    pd.DataFrame
        The sparsified DataFrame with NaN values.
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
    """
    Randomly flip a fraction of each row's entries to a different valid value.
    Valid values are inferred per column from the unique values in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    keep_fraction : float or array-like of shape (n_rows,)
        The fraction of entries to keep unchanged (between 0 and 1).
        If array-like, each element sets the keep fraction for the corresponding row.
    generator : np.random.Generator, optional
        A random number generator for reproducibility.
        Default is np.random.default_rng().

    Returns
    -------
    pd.DataFrame
        A new DataFrame with noisy values where flipped entries are replaced.
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
    """
    Scale the reaction values to a specified range.

    Parameters
    ----------
    reactions : np.ndarray
        The input array of reaction values to be scaled.
    min_value : float, optional
        The minimum value of the scaled range (default is 0.0).
    max_value : float, optional
        The maximum value of the scaled range (default is 1.0). 

    Returns
    -------
    np.ndarray
        The scaled reaction values.
    """
    scaled = (reactions - np.nanmin(reactions)) / (np.nanmax(reactions) - np.nanmin(reactions))
    return scaled * (max_value - min_value) + min_value

def binarize_reactions(
    array: np.ndarray,
    generator: np.random.Generator = np.random.default_rng(0),
) -> np.ndarray:
    """
    Binarize the input array based on random thresholds.

    Parameters
    ----------
    array : np.ndarray
        The input array to be binarized.
    generator : np.random.Generator, optional
        A random number generator for reproducibility.  
        Default is np.random.default_rng(0).

    Returns
    -------
    np.ndarray
        The binarized array.
    """
    random_matrix = generator.random(array.shape)
    return (random_matrix <= array).astype(int)


def add_ones(array: np.ndarray) -> np.ndarray:
    """
    Add a column of ones to the input array, typically for including an intercept term in linear models.
    
    Parameters
    ----------
    array : np.ndarray
        The input array to which a column of ones will be added.
        
    Returns
    -------
    np.ndarray
        The input array with an additional column of ones.
    """
    return np.hstack((array, np.ones((len(array),1))))


def create_meshgrid(
    limits: tuple[float, float, float, float],
    sampling_resolution: int,
) -> np.ndarray:
    """
    Create a meshgrid of points within the specified limits and sampling resolution.

    Parameters
    ----------
    limits : list or tuple of length 4
        The limits of the grid in the format [x_min, x_max, y_min, y_max].
    sampling_resolution : int
        The number of points to sample along each axis.

    Returns
    -------
    np.ndarray
        An array of shape (sampling_resolution*sampling_resolution, 2) containing the coordinates of the grid points.
    """
    xx, yy = np.meshgrid(np.linspace(limits[0], limits[1], sampling_resolution),
                            np.linspace(limits[2], limits[3], sampling_resolution))
    return np.c_[xx.ravel(), yy.ravel()]


def transformation_matrix(
    rotate: float = 0.0,
    scale: tuple[float, float] = (1.0, 1.0),
    shear: float = 0.0,
    order: tuple[Literal["shear", "rotate", "scale"], ...] = ("shear", "rotate", "scale"),
) -> np.ndarray:
    """
    Apply a 2D linear transformation to points.

    Parameters
    ----------
    rotate : float
        Rotation angle in degree (counterclockwise).
    scale : tuple of (sx, sy)
        Scaling factors along x and y axes.
    shear : float
        Shear factor in x-direction.
    order : tuple of str
        Order of transformations. Each can be 'scale', 'shear', 'rotate'.

    Returns
    -------
    matrix : np.ndarray, shape (2, 2)
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
    """
    Compute the Rasch model values for a given set of scores.

    Parameters
    ----------
    scores : np.array
        The input scores for which to compute the Rasch values.
    num_options : int, optional
        The number of answer options (default is 5).
    variance : float, optional
        The variance for the normal distribution used in the Rasch model (default is 0.1).
    
    Returns
    -------
    np.array
        The computed Rasch values for the input scores.
    np.array
        The answer options used in the Rasch model.
    """
    answer_options = np.linspace(0,1,num_options)
    if num_options == 2:
        return np.array([1 - scores, scores]), answer_options
    else:
        values = np.array([norm.pdf(scores, mu, variance) for mu in answer_options])
        return values / values.sum(axis=0), answer_options
    
def mean_impute(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with column means."""
    col_means = np.nanmean(X, axis=0)
    X_imputed = X.copy()
    nan_mask = np.isnan(X_imputed)
    X_imputed[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    return X_imputed


def normalize_embedding(embedding: np.ndarray, scaling: float = 1.05) -> np.ndarray:
    """Center and scale the embedding to fit within the defined limits."""
    assert embedding is not None, "Embedding must be initialized before normalizing."
    centroid = (embedding.max(axis=0) + embedding.min(axis=0))/2
    embedding -= centroid
    max_extent = np.abs(embedding).max(axis=0)*scaling
    return embedding / max_extent