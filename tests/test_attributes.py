from ixplore import IXPLORE

from conftest import D, G, K, N


def test_grid_shapes(xplore: IXPLORE) -> None:
    assert xplore.X.shape == (G, 2)
    assert xplore.X_transformed.shape == (G, D)
    assert xplore.log_prior_X.shape == (G,)
    assert xplore.n_features == D


def test_fitted_state_shapes(xplore: IXPLORE) -> None:
    assert xplore.embedding.shape == (N, 2)
    assert xplore.item_parameters.shape == (K, D + 1)
    assert xplore.predictions_X.shape == (G, K)
    assert xplore.log_likelihoods_X.shape == (N, G)
    assert len(xplore.parameter_names) == D + 1
