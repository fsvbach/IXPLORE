from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression


def sparsen(
    df: pd.DataFrame,
    keep_fraction: float,
    generator: np.random.Generator = np.random.default_rng(0),
) -> pd.DataFrame:
    """
    Randomly set a fraction of the entries in the DataFrame to NaN, simulating missing data.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be sparsified.
    keep_fraction : float
        The fraction of entries to keep (between 0 and 1).
    generator : np.random.Generator, optional
        A random number generator for reproducibility.  
        Default is np.random.default_rng(0).
    
    Returns
    -------
    pd.DataFrame
        The sparsified DataFrame with NaN values.
    """
    values = df.values
    mask = generator.random(values.shape) < keep_fraction
    sparse_array = np.where(mask, values, np.nan)
    return pd.DataFrame(sparse_array, index=df.index, columns=df.columns)


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

def extract_parameters(model: LogisticRegression) -> np.ndarray:
    """
    Extract the parameters from a fitted logistic regression model.

    Parameters
    ----------
    model : sklearn.linear_model.LogisticRegression
        The fitted logistic regression model. Assumes that the model has been fitted with fit_intercept=True.

    Returns
    -------
    np.ndarray
        The extracted parameters from the model.
    """
    model_params = np.concatenate((model.coef_, model.intercept_.reshape(1, 1)), axis=1)
    return model_params


def binarize(
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
    rotation: float = 0.0,
    scale: tuple[float, float] = (1.0, 1.0),
    shear: float = 0.0,
    order: tuple[Literal["shear", "rotate", "scale"], ...] = ("shear", "rotate", "scale"),
) -> np.ndarray:
    """
    Apply a 2D linear transformation to points.

    Parameters
    ----------
    rotation : float
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
    R = np.array([[np.cos(np.radians(rotation)), -np.sin(np.radians(rotation))],
                    [np.sin(np.radians(rotation)),  np.cos(np.radians(rotation))]])
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
    variance: float = 0.1,
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
        return np.array([scores, 1 - scores]), answer_options
    else:
        values = np.array([norm.pdf(scores, mu, variance) for mu in answer_options])
        return values / values.sum(axis=0), answer_options