"""Shared fixtures for the PyPhi refactor test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic numeric matrices for X and Y."""
    rng = np.random.default_rng(seed=42)
    X = rng.normal(size=(50, 10))
    Y = rng.normal(size=(50, 3))
    return X, Y


@pytest.fixture
def sample_dataframe(sample_data: tuple[np.ndarray, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return dataframes with observation identifiers."""
    X, Y = sample_data
    X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
    X_df.insert(0, "ObsID", [f"Obs{i}" for i in range(X.shape[0])])

    Y_df = pd.DataFrame(Y, columns=[f"Y{i}" for i in range(Y.shape[1])])
    Y_df.insert(0, "ObsID", [f"Obs{i}" for i in range(Y.shape[0])])
    return X_df, Y_df


@pytest.fixture
def missing_data() -> np.ndarray:
    """Return an array containing NaN values for missing-data tests."""
    rng = np.random.default_rng(seed=123)
    X = rng.normal(size=(30, 6))
    mask = rng.random(size=X.shape) < 0.15
    X[mask] = np.nan
    return X
