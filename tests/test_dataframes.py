import pandas as pd

from ixplore import IXPLORE

from conftest import D, G, K, N


def test_get_embedding(xplore: IXPLORE) -> None:
    df = xplore.get_embedding()
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (N, 2)
    assert df.columns.tolist() == ["x", "y"]
    assert df.index.equals(xplore.users)


def test_get_parameters(xplore: IXPLORE) -> None:
    df = xplore.get_parameters()
    assert df.shape == (K, D + 1)
    assert df.columns.tolist() == xplore.parameter_names
    assert df.index.equals(xplore.items)


def test_get_posteriors(xplore: IXPLORE) -> None:
    df = xplore.get_posteriors()
    assert df.shape == (N, G)
    assert df.index.equals(xplore.users)
