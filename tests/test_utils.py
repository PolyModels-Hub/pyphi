"""Tests for utility helpers in pyphi.utils."""

from __future__ import annotations

import numpy as np
from scipy.stats import f as f_dist

from pyphi import utils


# =============================================================================
# Tests for mean() and std()
# =============================================================================


def test_mean_ignores_nan(sample_data):
    """Test that mean() correctly ignores NaN values."""
    X, _ = sample_data
    X[0, 1] = np.nan
    X[4, 3] = np.nan
    result = utils.mean(X)
    expected = np.nanmean(X, axis=0, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_mean_without_nan(sample_data):
    """Test mean() on data without NaN values."""
    X, _ = sample_data
    result = utils.mean(X)
    expected = np.mean(X, axis=0, keepdims=True)
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_mean_all_nan_column():
    """Test mean() handles all-NaN columns gracefully."""
    X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    result = utils.mean(X)
    assert result.shape == (1, 2)
    np.testing.assert_allclose(result[0, 0], 2.0, rtol=1e-12)
    assert np.isnan(result[0, 1])


def test_std_matches_nanstd():
    """Test std() matches np.nanstd with ddof=1."""
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


def test_std_without_nan(sample_data):
    """Test std() on data without NaN values."""
    X, _ = sample_data
    result = utils.std(X)
    expected = np.std(X, axis=0, ddof=1, keepdims=True)
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
    """Test that invalid mcs flag returns original data unchanged."""
    X = np.array([[1.0], [3.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X, mcs="invalid")

    assert X_proc is X
    assert np.isnan(x_mean)
    assert np.isnan(x_std)


def test_meancenterscale_with_nan():
    """Test meancenterscale correctly handles NaN values."""
    X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    X_proc, x_mean, x_std = utils.meancenterscale(X)

    # Mean should ignore NaN
    expected_mean = np.nanmean(X, axis=0, keepdims=True)
    np.testing.assert_allclose(x_mean, expected_mean, rtol=1e-12)

    # Std should ignore NaN
    expected_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
    np.testing.assert_allclose(x_std, expected_std, rtol=1e-12)


# =============================================================================
# Tests for f95() and f99()
# =============================================================================


def test_f95_matches_scipy():
    """Test f95() matches scipy.stats.f.ppf(0.95, ...)."""
    # Test various degrees of freedom combinations
    test_cases = [
        (1, 10),
        (2, 20),
        (5, 50),
        (10, 100),
        (3, 7),
    ]
    for dfn, dfd in test_cases:
        result = utils.f95(dfn, dfd)
        expected = f_dist.ppf(0.95, dfn, dfd)
        np.testing.assert_allclose(
            result, expected, rtol=1e-12,
            err_msg=f"f95({dfn}, {dfd}) failed"
        )


def test_f99_matches_scipy():
    """Test f99() matches scipy.stats.f.ppf(0.99, ...)."""
    test_cases = [
        (1, 10),
        (2, 20),
        (5, 50),
        (10, 100),
        (3, 7),
    ]
    for dfn, dfd in test_cases:
        result = utils.f99(dfn, dfd)
        expected = f_dist.ppf(0.99, dfn, dfd)
        np.testing.assert_allclose(
            result, expected, rtol=1e-12,
            err_msg=f"f99({dfn}, {dfd}) failed"
        )


def test_f95_known_values():
    """Test f95() against known critical values."""
    # F(1, 10) at alpha=0.05 is approximately 4.965
    result = utils.f95(1, 10)
    assert 4.9 < result < 5.0

    # F(2, 20) at alpha=0.05 is approximately 3.493
    result = utils.f95(2, 20)
    assert 3.4 < result < 3.6


def test_f99_known_values():
    """Test f99() against known critical values."""
    # F(1, 10) at alpha=0.01 is approximately 10.044
    result = utils.f99(1, 10)
    assert 10.0 < result < 10.1

    # F(2, 20) at alpha=0.01 is approximately 5.849
    result = utils.f99(2, 20)
    assert 5.8 < result < 5.9


def test_f95_f99_relationship():
    """Test that f99 > f95 for same degrees of freedom."""
    test_cases = [(1, 10), (5, 50), (10, 100)]
    for dfn, dfd in test_cases:
        assert utils.f99(dfn, dfd) > utils.f95(dfn, dfd)


# =============================================================================
# Tests for n2z() and z2n()
# =============================================================================


def test_n2z_converts_nan_to_zero():
    """Test that n2z() converts NaN values to zeros."""
    X = np.array([[1.0, np.nan], [np.nan, 4.0]])
    X_result, nan_map = utils.n2z(X.copy())

    # NaN should be replaced with 0
    expected = np.array([[1.0, 0.0], [0.0, 4.0]])
    np.testing.assert_array_equal(X_result, expected)

    # nan_map should mark original NaN positions
    expected_map = np.array([[0, 1], [1, 0]])
    np.testing.assert_array_equal(nan_map, expected_map)


def test_n2z_no_nan():
    """Test n2z() when there are no NaN values."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    X_result, nan_map = utils.n2z(X.copy())

    np.testing.assert_array_equal(X_result, X)
    np.testing.assert_array_equal(nan_map, np.zeros_like(X, dtype=int))


def test_z2n_restores_nan():
    """Test that z2n() restores NaN values from a map."""
    X = np.array([[1.0, 0.0], [0.0, 4.0]])
    nan_map = np.array([[0, 1], [1, 0]])

    X_result = utils.z2n(X.copy(), nan_map)

    assert np.isnan(X_result[0, 1])
    assert np.isnan(X_result[1, 0])
    np.testing.assert_equal(X_result[0, 0], 1.0)
    np.testing.assert_equal(X_result[1, 1], 4.0)


def test_n2z_z2n_roundtrip():
    """Test that n2z followed by z2n restores original NaN positions."""
    X_original = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]])
    X = X_original.copy()

    X_zeroed, nan_map = utils.n2z(X)
    X_restored = utils.z2n(X_zeroed, nan_map)

    # Check NaN positions match
    np.testing.assert_array_equal(np.isnan(X_restored), np.isnan(X_original))


# =============================================================================
# Tests for spe_ci()
# =============================================================================


def test_spe_ci_returns_positive_limits():
    """Test that spe_ci() returns positive confidence limits."""
    spe = np.array([0.5, 1.2, 0.8, 1.5, 0.9, 1.1])
    lim95, lim99 = utils.spe_ci(spe)

    assert lim95 > 0
    assert lim99 > 0
    assert lim99 > lim95  # 99% limit should be higher


def test_spe_ci_near_zero_spe():
    """Test spe_ci() with near-zero SPE values."""
    spe = np.array([1e-20, 1e-20, 1e-20])
    lim95, lim99 = utils.spe_ci(spe)

    assert lim95 == 0.0
    assert lim99 == 0.0


def test_spe_ci_uses_chi2():
    """Test that spe_ci() correctly uses chi-squared distribution."""
    from scipy.stats import chi2

    spe = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lim95, lim99 = utils.spe_ci(spe)

    # Manual calculation
    spe_mean = np.mean(spe)
    spe_var = np.var(spe, ddof=1)
    g = spe_var / (2 * spe_mean)
    h = (2 * spe_mean**2) / spe_var

    expected_95 = g * chi2.ppf(0.95, h)
    expected_99 = g * chi2.ppf(0.99, h)

    np.testing.assert_allclose(lim95, expected_95, rtol=1e-10)
    np.testing.assert_allclose(lim99, expected_99, rtol=1e-10)


# =============================================================================
# Tests for single_score_conf_int()
# =============================================================================


def test_single_score_conf_int_returns_positive():
    """Test that single_score_conf_int() returns positive limits."""
    t = np.array([1.0, 2.0, -1.0, 0.5, -0.5])
    lim95, lim99 = utils.single_score_conf_int(t)

    assert lim95 > 0
    assert lim99 > 0
    assert lim99 > lim95


def test_single_score_conf_int_uses_t_dist():
    """Test that single_score_conf_int() correctly uses t-distribution."""
    from scipy.stats import t as t_dist

    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    lim95, lim99 = utils.single_score_conf_int(scores)

    n = scores.shape[0]
    st = np.var(scores, ddof=1)

    expected_95 = t_dist.ppf(0.975, n - 1) * np.sqrt(st)
    expected_99 = t_dist.ppf(0.995, n - 1) * np.sqrt(st)

    np.testing.assert_allclose(lim95, expected_95, rtol=1e-10)
    np.testing.assert_allclose(lim99, expected_99, rtol=1e-10)


def test_single_score_conf_int_sample_size_effect():
    """Test that larger samples give narrower confidence intervals."""
    np.random.seed(42)
    small_sample = np.random.randn(10)
    large_sample = np.random.randn(100)

    # Normalize to same variance for fair comparison
    small_sample = small_sample / np.std(small_sample)
    large_sample = large_sample / np.std(large_sample)

    lim95_small, _ = utils.single_score_conf_int(small_sample)
    lim95_large, _ = utils.single_score_conf_int(large_sample)

    # Larger sample should give narrower CI (smaller limit relative to std)
    assert lim95_large < lim95_small


# =============================================================================
# Tests for find()
# =============================================================================


def test_find_basic():
    """Test basic find() functionality."""
    a = np.array([1, 5, 3, 8, 2])
    result = utils.find(a, lambda x: x > 3)

    assert result == [1, 3]  # indices of 5 and 8


def test_find_no_matches():
    """Test find() when no elements match."""
    a = np.array([1, 2, 3])
    result = utils.find(a, lambda x: x > 10)

    assert result == []


def test_find_all_match():
    """Test find() when all elements match."""
    a = np.array([5, 6, 7])
    result = utils.find(a, lambda x: x > 0)

    assert result == [0, 1, 2]


def test_find_with_equality():
    """Test find() with equality condition."""
    a = np.array([1, 2, 2, 3, 2])
    result = utils.find(a, lambda x: x == 2)

    assert result == [1, 2, 4]


# =============================================================================
# Tests for unique()
# =============================================================================


def test_unique_preserves_order():
    """Test that unique() preserves order of first occurrence."""
    import pandas as pd

    df = pd.DataFrame({"A": [3, 1, 2, 1, 3, 2, 4]})
    result = utils.unique(df, "A")

    assert result == [3, 1, 2, 4]  # Order of first occurrence


def test_unique_single_value():
    """Test unique() with a single unique value."""
    import pandas as pd

    df = pd.DataFrame({"A": [5, 5, 5, 5]})
    result = utils.unique(df, "A")

    assert result == [5]


def test_unique_all_different():
    """Test unique() when all values are different."""
    import pandas as pd

    df = pd.DataFrame({"A": [1, 2, 3, 4]})
    result = utils.unique(df, "A")

    assert result == [1, 2, 3, 4]


def test_unique_with_strings():
    """Test unique() with string values."""
    import pandas as pd

    df = pd.DataFrame({"name": ["bob", "alice", "bob", "charlie", "alice"]})
    result = utils.unique(df, "name")

    assert result == ["bob", "alice", "charlie"]
