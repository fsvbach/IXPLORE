import numpy as np
import pandas as pd
import pytest

from ixplore import IXPLORE
from ixplore.posteriors import posterior_maps

from conftest import D, G, K, N, SAMPLING_RESOLUTION


@pytest.fixture
def fresh(data: pd.DataFrame) -> IXPLORE:
    model = IXPLORE(data, sampling_resolution=SAMPLING_RESOLUTION)
    model.fit_posteriors()
    return model


def test_iterate(fresh: IXPLORE) -> None:
    fresh.iterate(n_iterations=2)
    assert fresh.embedding.shape == (N, 2)
    assert fresh.item_parameters.shape == (K, D + 1)
    assert fresh.log_likelihoods_X.shape == (N, G)


def test_apply_prior(fresh: IXPLORE) -> None:
    fresh.apply_prior(prior_variance=0.5)
    assert fresh.embedding.shape == (N, 2)
    assert fresh.log_prior_X.shape == (G,)


def test_fit_models_with_posteriors(fresh: IXPLORE) -> None:
    fresh.fit_models(point_estimates=False)
    assert fresh.item_parameters.shape == (K, D + 1)
    assert fresh._log_p_X.shape == (G, K)


def test_custom_kernel(data: pd.DataFrame) -> None:
    model = IXPLORE(
        data,
        sampling_resolution=SAMPLING_RESOLUTION,
        kernel=lambda X: np.hstack([X, X ** 2]),
    )
    assert model.X_transformed.shape == (G, 4)
    assert model.n_features == 4
    assert model.item_parameters.shape == (K, 5)
    assert len(model.parameter_names) == 5


def test_map_estimator(fresh: IXPLORE) -> None:
    fresh.get_point_estimates = posterior_maps
    fresh.fit_posteriors()
    assert fresh.embedding.shape == (N, 2)
    # MAP output must land on grid points
    grid_set = {tuple(row) for row in fresh.X}
    for row in fresh.embedding:
        assert tuple(row) in grid_set
