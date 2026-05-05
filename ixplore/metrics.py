"""Metrics for fitted IXPLORE models.

Two groups:
- Response-space scoring primitives (`compute_mae`, `compute_accuracy`,
  `compute_rmse`) over predicted-vs-true answer cells.
- Latent-geometry diagnostics (`compute_spread`, `compute_distortion`) over
  fitted embeddings and round-trip behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import utils


# ---------------------------------------------------------------------------
# Response-space metrics
# ---------------------------------------------------------------------------


def compute_mae(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    return float(np.mean(np.abs(true_vals - pred_vals)))


def compute_accuracy(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    return float(1 - np.mean(np.abs(np.round(true_vals) - np.round(pred_vals))))


def compute_rmse(true_vals: np.ndarray, pred_vals: np.ndarray) -> float:
    return float(np.sqrt(np.mean((true_vals - pred_vals) ** 2)))


# ---------------------------------------------------------------------------
# Latent-geometry metrics
# ---------------------------------------------------------------------------


def compute_spread(
    embedding: np.ndarray,
    limits: tuple[float, float],
    threshold: float = 0.05,
) -> dict[str, float]:
    """Spread of an embedding and the fraction of points near the grid edge.

    Parameters
    ----------
    embedding : (N, 2)
        User positions.
    limits : (lo, hi)
        Per-axis grid limits.
    threshold : float
        Margin for "near border", expressed as a fraction of the axis range.

    Returns
    -------
    {"spread", "boundary"}
        spread   : sqrt(var_x + var_y).
        boundary : fraction of points within `threshold * (hi - lo)` of any edge.
    """
    lo, hi = limits
    margin = (hi - lo) * threshold
    near_border = (
        (embedding[:, 0] - lo < margin)
        | (hi - embedding[:, 0] < margin)
        | (embedding[:, 1] - lo < margin)
        | (hi - embedding[:, 1] < margin)
    )
    spread = float(np.sqrt(embedding[:, 0].std() ** 2 + embedding[:, 1].std() ** 2))
    return {"spread": round(spread, 4), "boundary": round(float(near_border.mean()), 4)}


def compute_distortion(
    model,
    resolution: int = 30,
    sparsity: float | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Sample synthetic users on a grid, re-embed, and report displacement stats.

    Builds a `resolution × resolution` grid spanning `model.limits`, draws
    response probabilities from the fitted item models, optionally sparsifies
    them, and re-embeds each synthetic user via `model.embed`.

    Parameters
    ----------
    model : IXPLORE
        Fitted model.
    resolution : int
        Grid points per axis.
    sparsity : float | None
        If not None, fraction of items KEPT per synthetic user (in (0, 1]).
        None ⇒ full responses.
    random_state : int
        Seed for the sparsity mask.

    Returns
    -------
    grid_positions : (G, 2) array of true positions
    reembedded_positions : (G, 2) array of inferred positions
    displacement_mean : mean of ‖inferred − true‖
    displacement_std  : std of ‖inferred − true‖
    """
    grid = utils.create_meshgrid(model.limits, model.limits, resolution)
    responses = pd.DataFrame(model.predict(grid), columns=model.items)

    if sparsity is not None:
        rng = np.random.default_rng(random_state)
        responses = utils.add_sparsity(responses, keep_fraction=sparsity, generator=rng)

    reembedded = model.embed(responses)

    displacement = np.linalg.norm(reembedded - grid, axis=1)
    return grid, reembedded, float(displacement.mean()), float(displacement.std())
