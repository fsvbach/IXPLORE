import numpy as np
import pandas as pd
import pytest
from ixplore import IXPLORE

def test_ixplore_runs():
    """Test that IXPLORE runs without crashing on toy data."""
    X = pd.DataFrame(np.random.rand(10, 5))       # 10 points, 5 features
    xplore = IXPLORE(X)
    embedding = xplore.embedding
    # Check embedding shape
    assert embedding.shape == (10, 2), "Embedding shape should be (n_samples, 2)"