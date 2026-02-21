from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib import axes, rcParams
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.figure import Figure, SubFigure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .logger import logger

if TYPE_CHECKING:
    from .algorithm import IXPLORE

plt.rcParams.update({
    "figure.figsize": (5, 5),
    "figure.dpi": 300,
    "axes.labelsize": 7, 
    "font.size": 7,
    "legend.fontsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    'lines.linewidth': 1,
    "axes.facecolor":'None',
    'axes.titlesize': 7,
    'axes.titlepad' : 1,    
    'axes.linewidth': 0.5})

# Define the colors
pruple_hex = '#0127A4'
blue_hex = '#7696FE'
red_hex = '#DC6025'
orange_hex = '#EAA07D'
neutral_color = '#D9E1E8'

# Create custom colormap
colors = [blue_hex, neutral_color, orange_hex]
n_bins = 10  # Number of bins for levels
cmap_name = 'custom_cmap'
colormap: Colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


def figure(ax: axes.Axes | None = None) -> tuple[Figure | SubFigure, axes.Axes]:
    if not ax:
        fig, ax = plt.subplots()
        return fig, ax
    else:
        return ax.figure, ax


def clean_axis(ax: axes.Axes) -> axes.Axes:
    ax.set(aspect='equal',
           xticks=[],
           yticks=[])
    return ax


def plot_embedding(
    embedding: pd.DataFrame,
    colors: np.ndarray | str = 'gray',
    ax: axes.Axes | None = None,
    user: str | None = None,
    highlight: dict[str, Any] | None = None,
    **kwargs: Any,
) -> axes.Axes:
    if highlight is None:
        highlight = {}
    if isinstance(colors, str):
        colors = np.array([colors] * embedding.shape[0])

    fig, ax = figure(ax)

    scatter_kwargs: dict[str, Any] = {"zorder": 2, "edgecolors": "black", "s": 40, "lw": 0.5}
    scatter_kwargs.update(**kwargs)
    ax.scatter(embedding.loc[:, 'x'], 
               embedding.loc[:, 'y'], 
               c=colors, 
               **scatter_kwargs)

    if user is not None:
        params: dict[str, Any] = {'edgecolor': 'white', 's':7, 'color':'None', 'lw':1, 'zorder':5, 'label':f"User {user}"}
        params.update(highlight)
        # Put the highlighted user in front of all other users
        color = colors[embedding.index.get_loc(user)]
        ax.scatter(embedding.loc[user, 'x'], embedding.loc[user, 'y'], color=color, **scatter_kwargs)
        # Then add the highlight on top of that
        ax.scatter(embedding.loc[user, 'x'], embedding.loc[user, 'y'], **params)
    return ax


def plot_likelihood(
    xplore: IXPLORE,
    feature: str,
    cmap: Colormap = colormap,
    ax: axes.Axes | None = None,
) -> tuple[Figure | SubFigure, axes.Axes]:
    fig, ax = figure(ax)

    meshgrid_size = int(np.sqrt(xplore.X.shape[0]))
    xx = xplore.X[:, 0].reshape(meshgrid_size, meshgrid_size)
    yy = xplore.X[:, 1].reshape(meshgrid_size, meshgrid_size)

    # Predict probabilities on the grid
    assert xplore.likelihood_X is not None, "Likelihoods must be computed before plotting."
    Z = xplore.likelihood_X[:,xplore.items.get_loc(feature)]
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary at 50% probability
    ax.contour(xx, yy, Z, levels=[0.5], colors="black", linestyles="--")
    contour = ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap, levels=np.linspace(0, 1, 11), zorder=1)
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Likelihood')
    return fig, ax


def plot_posterior(
    xplore: IXPLORE,
    answers: pd.Series,
    cmap: Colormap = colormap,
    ax: axes.Axes | None = None,
    add_bar: bool = True,
) -> tuple[Figure | SubFigure, axes.Axes]:
    fig, ax = figure(ax)

    meshgrid_size = int(np.sqrt(xplore.X.shape[0]))
    xx = xplore.X[:, 0].reshape(meshgrid_size, meshgrid_size)
    yy = xplore.X[:, 1].reshape(meshgrid_size, meshgrid_size)

    # Predict probabilities on the grid
    Z = xplore.posterior_X(answers)
    zz = Z.reshape(xx.shape)

    # Plot heatmap of probability
    if add_bar:
        contour = ax.contourf(xx, yy, zz, alpha=0.8, cmap=cmap, zorder=1)
        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Probability')
        cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    # Indicate point estimate
    x, y = xplore.posteriors2coordinates(Z)[0]
    ax.scatter(x, y, marker='x', color='black', s=10, label='Optimized Coordinates', zorder=5)

    return fig, ax


def plot_overview(
    xplore: IXPLORE,
    question: str,
    user: str | None = None,
    colors: np.ndarray | str = 'gray',
    cmap: Colormap = plt.colormaps['viridis'],
    figsize: tuple[float, float] = (7, 4),
) -> tuple[Figure | SubFigure, tuple[axes.Axes, axes.Axes]]:
    """
    Plot an overview of the posterior distribution for a user and the likelihood for a question, along with the embedding.
    
    Parameters
    ----------
    xplore : IXPLORE
        The IXPLORE model instance containing the data and methods for computing the posterior and likelihood.
    question : str
        The question for which to plot the likelihood.
    user : str
        The user for whom to plot the posterior distribution. If None, the posterior plot will be skipped. Default is None.
    colors : array-like or str, optional
        The colors to use for the embedding points. Can be a single color or an array of colors corresponding to each point. Default is 'black'.
    cmap : Colormap, optional
        The size of the figure in inches. Default is (7, 4).
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)

    plot_embedding(xplore.get_embedding(), colors=colors, s=60,
                   user=user, highlight={'s': 60, 'edgecolor': 'white'},
                   ax=ax1)
    
    # Plot posterior for the user if specified
    if user is not None:
        i = xplore.users.get_loc(user)
        n_answers = xplore.reactions[i, :]
        user_answers = pd.Series(n_answers, index=xplore.items, name=user).dropna()
        plot_posterior(xplore, user_answers, ax=ax1)
        ax1.set_title(f'Posterior for User {user}')

    i = xplore.items.get_loc(question)
    q_answers = np.array(list(map(cmap, xplore.reactions[:,i].astype(float))))
    plot_likelihood(xplore, question, ax=ax2)
    plot_embedding(xplore.get_embedding(), colors=q_answers, s=60,
                   user=user, highlight={'s': 60, 'edgecolor': 'white'},
                   ax=ax2)
    ax2.set_title(f'Likelihood for Question {question}')

    # Remove axis ticks and labels
    for ax in (ax1, ax2):
        clean_axis(ax)

    mae, acc = xplore.evaluate()
    logger.info(f'MAE: {mae:.4f}, ACC: {acc:.4f}')

    return fig, (ax1, ax2)