import numpy as np
import pandas as pd
import pytest

from ixplore import IXPLORE

from conftest import G, K

M = 3
S = 7

@pytest.fixture
def answers(xplore: IXPLORE) -> pd.Series:
    return pd.Series([0.5, 1.0, 0.0], index=xplore.items[:3])


def test_predict_full(xplore: IXPLORE) -> None:
    positions = np.zeros((M, 2))
    assert xplore.predict(positions).shape == (M, K)


def test_predict_subset(xplore: IXPLORE) -> None:
    positions = np.zeros((M, 2))
    assert xplore.predict(positions, items=xplore.items[:2]).shape == (M, 2)


def test_predict_empty(xplore: IXPLORE) -> None:
    assert xplore.predict(np.empty((0, 2))).shape == (0,)


def test_compute_posteriors_single(xplore: IXPLORE, answers: pd.Series) -> None:
    assert xplore.compute_posteriors(answers).shape == (G,)


def test_compute_posteriors_batch(xplore: IXPLORE, answers: pd.Series) -> None:
    batch = pd.DataFrame([answers, answers, answers])
    assert xplore.compute_posteriors(batch).shape == (3, G)


def test_embed_users_single(xplore: IXPLORE, answers: pd.Series) -> None:
    assert xplore.embed(answers).shape == (2,)


def test_embed_users_batch(xplore: IXPLORE, answers: pd.Series) -> None:
    batch = pd.DataFrame([answers, answers, answers])
    assert xplore.embed(batch).shape == (3, 2)


def test_impute_answers_single(xplore: IXPLORE, answers: pd.Series) -> None:
    out = xplore.impute_answers(answers)
    assert isinstance(out, pd.Series)
    assert len(out) == K
    assert out.index.equals(xplore.items)


def test_impute_answers_batch(xplore: IXPLORE, answers: pd.Series) -> None:
    batch = pd.DataFrame([answers, answers, answers])
    out = xplore.impute_answers(batch)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (3, K)
    assert out.columns.equals(xplore.items)


@pytest.mark.parametrize("method", ["rasch", "posterior", "random"])
def test_sample_answers(xplore: IXPLORE, answers: pd.Series, method: str) -> None:
    assert xplore.sample_answers(answers, method=method, num_samples=S).shape == (S, K)
