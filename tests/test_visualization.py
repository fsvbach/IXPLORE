import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ixplore import IXPLORE
from ixplore.visualization import (
    plot_embedding,
    plot_predictions,
    plot_posterior,
    plot_overview,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def answers(xplore: IXPLORE) -> pd.Series:
    return pd.Series([0.5, 1.0, 0.0], index=xplore.items[:3])


def test_plot_embedding(xplore: IXPLORE) -> None:
    plot_embedding(xplore.get_embedding())


def test_plot_embedding_with_user(xplore: IXPLORE) -> None:
    plot_embedding(xplore.get_embedding(), user=xplore.users[0])


def test_plot_predictions(xplore: IXPLORE) -> None:
    plot_predictions(xplore, feature=xplore.items[0])


def test_plot_posterior(xplore: IXPLORE, answers: pd.Series) -> None:
    plot_posterior(xplore, answers)


def test_plot_overview(xplore: IXPLORE) -> None:
    plot_overview(xplore, question=xplore.items[0], user=xplore.users[0])


def test_plot_overview_no_user(xplore: IXPLORE) -> None:
    plot_overview(xplore, question=xplore.items[0], user=None)
