"""Tests for duplicated utility helpers in pyphi.utils."""

from __future__ import annotations

import numpy as np

from pyphi import utils


def test_mean_ignores_nan(sample_data):
    X, _ = sample_data
    X[0, 1] = np.nan
    X[4, 3] = np.nan
    result = utils.mean(X)
    expected = np.nanmean(X, axis=0, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_std_matches_nanstd():
    X = np.array(
        [
            [1.0, 2.0, np.nan],
            [3.0, 4.0, 6.0],
            [5.0, np.nan, 8.0],
        ]
    )
    result = utils.std(X)
    expected = np.nanstd(X, axis=0, ddof=1, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_meancenterscale_with_default():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X)

    np.testing.assert_allclose(x_mean, np.array([[3.0, 4.0]]))
    np.testing.assert_allclose(x_std, np.array([[2.0, 2.0]]))
    np.testing.assert_allclose(np.mean(X_proc, axis=0), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(np.std(X_proc, axis=0, ddof=1), np.ones(2), atol=1e-12)


def test_meancenterscale_center_only():
    X = np.array([[2.0, 4.0], [6.0, 8.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X, mcs="center")

    np.testing.assert_allclose(x_mean, utils.mean(X))
    np.testing.assert_allclose(x_std, np.ones((1, X.shape[1])))
    np.testing.assert_allclose(np.mean(X_proc, axis=0), np.zeros(2), atol=1e-12)


def test_meancenterscale_autoscale_only():
    X = np.array([[2.0, 4.0], [6.0, 8.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X, mcs="autoscale")

    np.testing.assert_allclose(x_mean, np.zeros((1, X.shape[1])))
    np.testing.assert_allclose(x_std, utils.std(X))
    np.testing.assert_allclose(np.std(X_proc, axis=0, ddof=1), np.ones(2), atol=1e-12)


def test_meancenterscale_returns_original_when_disabled():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X, mcs=False)

    assert X_proc is X
    assert np.isnan(x_mean)
    assert np.isnan(x_std)


def test_meancenterscale_invalid_flag():
    X = np.array([[1.0], [3.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X, mcs="invalid")

    assert X_proc is X
    assert np.isnan(x_mean)
    assert np.isnan(x_std)
