import numpy as np
import pandas as pd
import pytest

from ixplore import IXPLORE

N = 8
K = 5
SAMPLING_RESOLUTION = 20
G = SAMPLING_RESOLUTION ** 2
D = 2  # identity kernel


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.random((N, K)))


@pytest.fixture(scope="session")
def xplore(data: pd.DataFrame) -> IXPLORE:
    model = IXPLORE(data, sampling_resolution=SAMPLING_RESOLUTION)
    model.fit_posteriors()
    return model
