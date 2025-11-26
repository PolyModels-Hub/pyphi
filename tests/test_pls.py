"""Tests for the PLS module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyphi import pls, pls_, pls_pred, spe, prep_pls_4_MDbyNLP, cca, cca_multi


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def complete_data():
    """Generate complete X and Y data (no missing values) for PLS testing."""
    np.random.seed(42)
    n_samples, n_features, n_responses = 50, 10, 3
    X = np.random.randn(n_samples, n_features)
    # Create Y correlated with X for meaningful PLS
    Y = X[:, :3] @ np.random.randn(3, n_responses) + 0.1 * np.random.randn(
        n_samples, n_responses
    )
    return X, Y


@pytest.fixture
def complete_data_univariate():
    """Generate data with univariate Y."""
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    Y = X[:, :3].sum(axis=1, keepdims=True) + 0.1 * np.random.randn(n_samples, 1)
    return X, Y


@pytest.fixture
def data_with_missing():
    """Generate data with missing values for PLS testing."""
    np.random.seed(42)
    n_samples, n_features, n_responses = 50, 10, 2
    X = np.random.randn(n_samples, n_features)
    Y = X[:, :2] @ np.random.randn(2, n_responses) + 0.1 * np.random.randn(
        n_samples, n_responses
    )
    # Introduce ~5% missing data in X
    X[0, 1] = np.nan
    X[5, 3] = np.nan
    X[10, 7] = np.nan
    X[15, 2] = np.nan
    X[20, 9] = np.nan
    return X, Y


@pytest.fixture
def dataframe_data():
    """Generate DataFrames with observation IDs for PLS testing."""
    np.random.seed(42)
    n_samples, n_features, n_responses = 30, 8, 2
    X = np.random.randn(n_samples, n_features)
    Y = X[:, :2] @ np.random.randn(2, n_responses) + 0.1 * np.random.randn(
        n_samples, n_responses
    )
    obs_ids = [f"Obs_{i}" for i in range(n_samples)]
    var_ids_x = [f"X{j}" for j in range(n_features)]
    var_ids_y = [f"Y{j}" for j in range(n_responses)]
    df_x = pd.DataFrame(X, columns=var_ids_x)
    df_x.insert(0, "ObsID", obs_ids)
    df_y = pd.DataFrame(Y, columns=var_ids_y)
    df_y.insert(0, "ObsID", obs_ids)
    return df_x, df_y


# =============================================================================
# Basic PLS Tests
# =============================================================================


class TestPLSBasic:
    """Test basic PLS functionality."""

    def test_pls_returns_dict(self, complete_data):
        """Test that pls() returns a dictionary."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert isinstance(model, dict)

    def test_pls_contains_required_keys(self, complete_data):
        """Test that PLS model contains all required keys."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        required_keys = [
            "T", "P", "Q", "W", "Ws", "U",
            "r2x", "r2xpv", "mx", "sx",
            "r2y", "r2ypv", "my", "sy",
            "T2", "speX", "speY", "type",
        ]
        for key in required_keys:
            assert key in model, f"Missing key: {key}"

    def test_pls_scores_shape(self, complete_data):
        """Test that scores matrix T has correct shape."""
        X, Y = complete_data
        n_components = 3
        model = pls(X, Y, n_components, shush=True)
        assert model["T"].shape == (X.shape[0], n_components)

    def test_pls_loadings_shape(self, complete_data):
        """Test that loadings matrices have correct shapes."""
        X, Y = complete_data
        n_components = 3
        model = pls(X, Y, n_components, shush=True)
        assert model["P"].shape == (X.shape[1], n_components)
        assert model["Q"].shape == (Y.shape[1], n_components)

    def test_pls_weights_shape(self, complete_data):
        """Test that weight matrices have correct shapes."""
        X, Y = complete_data
        n_components = 3
        model = pls(X, Y, n_components, shush=True)
        assert model["W"].shape == (X.shape[1], n_components)
        assert model["Ws"].shape == (X.shape[1], n_components)

    def test_pls_r2x_is_positive(self, complete_data):
        """Test that R2X values are positive."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert np.all(model["r2x"] >= 0)

    def test_pls_r2y_is_positive(self, complete_data):
        """Test that R2Y values are positive."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert np.all(model["r2y"] >= 0)

    def test_pls_r2x_sums_less_than_one(self, complete_data):
        """Test that cumulative R2X is <= 1."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert np.sum(model["r2x"]) <= 1.0 + 1e-10

    def test_pls_r2y_sums_less_than_one(self, complete_data):
        """Test that cumulative R2Y is <= 1."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert np.sum(model["r2y"]) <= 1.0 + 1e-10

    def test_pls_type_is_pls(self, complete_data):
        """Test that model type is 'pls'."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert model["type"] == "pls"

    def test_pls_univariate_y(self, complete_data_univariate):
        """Test PLS with univariate Y."""
        X, Y = complete_data_univariate
        model = pls(X, Y, 2, shush=True)
        assert model["Q"].shape == (1, 2)
        assert model["r2y"].shape == (2,) or np.isscalar(model["r2y"]) or model["r2y"].size == 2


# =============================================================================
# PLS Algorithm Tests
# =============================================================================


class TestPLSAlgorithms:
    """Test different PLS algorithms."""

    def test_pls_nipals_default(self, complete_data):
        """Test that NIPALS is used by default with force_nipals=True."""
        X, Y = complete_data
        # This should work without errors
        model = pls(X, Y, 3, force_nipals=True, shush=True)
        assert "T" in model

    def test_pls_svd_when_complete(self, complete_data):
        """Test that SVD can be used for complete data."""
        X, Y = complete_data
        model = pls(X, Y, 3, force_nipals=False, shush=True)
        # SVD or NIPALS - both should produce valid model
        assert "T" in model
        assert np.all(np.isfinite(model["T"]))

    def test_pls_with_missing_data(self, data_with_missing):
        """Test PLS handles missing data."""
        X, Y = data_with_missing
        model = pls(X, Y, 2, shush=True)
        assert "T" in model
        # Scores should not have NaN
        assert np.all(np.isfinite(model["T"]))


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestPLSPreprocessing:
    """Test preprocessing options."""

    def test_pls_mcs_true(self, complete_data):
        """Test mean-center and autoscale preprocessing."""
        X, Y = complete_data
        model = pls(X, Y, 2, mcsX=True, mcsY=True, shush=True)
        # Check that preprocessing was applied
        assert model["mx"].shape[1] == X.shape[1]
        assert model["sx"].shape[1] == X.shape[1]
        assert model["my"].shape[1] == Y.shape[1]
        assert model["sy"].shape[1] == Y.shape[1]

    def test_pls_mcs_false(self, complete_data):
        """Test no preprocessing."""
        X, Y = complete_data
        model = pls(X, Y, 2, mcsX=False, mcsY=False, shush=True)
        # Mean should be zeros, std should be ones
        assert np.allclose(model["mx"], 0)
        assert np.allclose(model["sx"], 1)
        assert np.allclose(model["my"], 0)
        assert np.allclose(model["sy"], 1)

    def test_pls_mcs_center(self, complete_data):
        """Test center-only preprocessing."""
        X, Y = complete_data
        model = pls(X, Y, 2, mcsX="center", mcsY="center", shush=True)
        # std should be ones (no scaling)
        assert np.allclose(model["sx"], 1)
        assert np.allclose(model["sy"], 1)

    def test_pls_mcs_autoscale(self, complete_data):
        """Test autoscale-only preprocessing."""
        X, Y = complete_data
        model = pls(X, Y, 2, mcsX="autoscale", mcsY="autoscale", shush=True)
        # mean should be zeros (no centering)
        assert np.allclose(model["mx"], 0)
        assert np.allclose(model["my"], 0)


# =============================================================================
# DataFrame Support Tests
# =============================================================================


class TestPLSDataFrame:
    """Test DataFrame support."""

    def test_pls_with_dataframe(self, dataframe_data):
        """Test PLS with DataFrame input."""
        df_x, df_y = dataframe_data
        model = pls(df_x, df_y, 2, shush=True)
        assert "obsidX" in model
        assert "varidX" in model
        assert "obsidY" in model
        assert "varidY" in model

    def test_pls_dataframe_preserves_ids(self, dataframe_data):
        """Test that observation/variable IDs are preserved."""
        df_x, df_y = dataframe_data
        model = pls(df_x, df_y, 2, shush=True)
        assert len(model["obsidX"]) == df_x.shape[0]
        assert len(model["varidX"]) == df_x.shape[1] - 1  # Minus ObsID column
        assert model["varidX"][0] == "X0"


# =============================================================================
# PLS Prediction Tests
# =============================================================================


class TestPLSPrediction:
    """Test PLS prediction functionality."""

    def test_pls_pred_returns_dict(self, complete_data):
        """Test that pls_pred returns a dictionary."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        pred = pls_pred(X, model)
        assert isinstance(pred, dict)

    def test_pls_pred_contains_required_keys(self, complete_data):
        """Test that prediction contains required keys."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        pred = pls_pred(X, model)
        required_keys = ["Yhat", "Xhat", "Tnew", "speX", "T2"]
        for key in required_keys:
            assert key in pred, f"Missing key: {key}"

    def test_pls_pred_yhat_shape(self, complete_data):
        """Test that Yhat has correct shape."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        pred = pls_pred(X, model)
        assert pred["Yhat"].shape == Y.shape

    def test_pls_pred_xhat_shape(self, complete_data):
        """Test that Xhat has correct shape."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        pred = pls_pred(X, model)
        assert pred["Xhat"].shape == X.shape

    def test_pls_pred_tnew_shape(self, complete_data):
        """Test that Tnew has correct shape."""
        X, Y = complete_data
        n_components = 3
        model = pls(X, Y, n_components, shush=True)
        pred = pls_pred(X, model)
        assert pred["Tnew"].shape == (X.shape[0], n_components)

    def test_pls_pred_on_new_data(self, complete_data):
        """Test prediction on new data."""
        X, Y = complete_data
        model = pls(X[:40], Y[:40], 3, shush=True)
        pred = pls_pred(X[40:], model)
        assert pred["Yhat"].shape == (10, Y.shape[1])

    def test_pls_pred_single_row(self, complete_data):
        """Test prediction on single row."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        pred = pls_pred(X[0], model)
        assert pred["Yhat"].shape[0] == 1

    def test_pls_pred_with_missing_data(self, data_with_missing):
        """Test prediction with missing data in new observations."""
        X, Y = data_with_missing
        model = pls(X, Y, 2, shush=True)
        # Create new data with missing values
        X_new = X[:5].copy()
        X_new[0, 0] = np.nan
        pred = pls_pred(X_new, model)
        assert np.all(np.isfinite(pred["Yhat"]))


# =============================================================================
# SPE Tests
# =============================================================================


class TestSPE:
    """Test SPE function."""

    def test_spe_returns_array(self, complete_data):
        """Test that spe() returns array(s)."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        spex = spe(model, X)
        assert isinstance(spex, np.ndarray)

    def test_spe_shape(self, complete_data):
        """Test SPE shape."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        spex = spe(model, X)
        assert spex.shape[0] == X.shape[0]

    def test_spe_with_ynew(self, complete_data):
        """Test SPE with Y provided."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        spex, spey = spe(model, X, Ynew=Y)
        assert spex.shape[0] == X.shape[0]
        assert spey.shape[0] == Y.shape[0]

    def test_spe_positive_values(self, complete_data):
        """Test that SPE values are non-negative."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        spex = spe(model, X)
        assert np.all(spex >= 0)


# =============================================================================
# CCA Tests
# =============================================================================


class TestCCA:
    """Test CCA functions."""

    def test_cca_returns_tuple(self, complete_data):
        """Test that cca() returns tuple."""
        X, Y = complete_data
        result = cca(X, Y[:, :1])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_cca_correlation_value(self, complete_data):
        """Test that CCA correlation is valid."""
        X, Y = complete_data
        corr, wx, wy = cca(X, Y[:, :1])
        # Correlation should be a scalar
        assert np.isscalar(corr) or corr.size == 1

    def test_cca_weights_shapes(self, complete_data):
        """Test CCA weight vector shapes."""
        X, Y = complete_data
        corr, wx, wy = cca(X, Y[:, :1])
        assert wx.shape[0] == X.shape[1]
        assert wy.shape[0] == 1

    def test_cca_multi_returns_dict(self, complete_data):
        """Test that cca_multi() returns dict."""
        X, Y = complete_data
        result = cca_multi(X, Y, num_components=2)
        assert isinstance(result, dict)

    def test_cca_multi_contains_keys(self, complete_data):
        """Test cca_multi() contains expected keys."""
        X, Y = complete_data
        result = cca_multi(X, Y, num_components=2)
        assert "correlations" in result
        assert "W_X" in result
        assert "W_Y" in result


# =============================================================================
# Cross-Validation Tests
# =============================================================================


class TestPLSCrossValidation:
    """Test PLS cross-validation."""

    def test_pls_with_cross_val(self, complete_data):
        """Test PLS with cross-validation."""
        X, Y = complete_data
        model = pls(X, Y, 2, cross_val=10, shush=True)
        assert "q2Y" in model
        assert "q2Ypv" in model

    def test_pls_cross_val_q2_values(self, complete_data):
        """Test that Q2Y values are reasonable."""
        X, Y = complete_data
        model = pls(X, Y, 2, cross_val=10, shush=True)
        # Q2 should be <= R2 (typically)
        # Just check it's a valid number
        if np.isscalar(model["q2Y"]):
            assert np.isfinite(model["q2Y"])
        else:
            assert np.all(np.isfinite(model["q2Y"]))

    def test_pls_cross_val_x(self, complete_data):
        """Test cross-validation with X."""
        X, Y = complete_data
        model = pls(X, Y, 2, cross_val=10, cross_val_X=True, shush=True)
        assert "q2X" in model
        assert "q2Xpv" in model

    def test_pls_leave_one_out(self, complete_data):
        """Test leave-one-out cross-validation."""
        X, Y = complete_data
        # Using smaller dataset for speed
        model = pls(X[:20], Y[:20], 2, cross_val=100, shush=True)
        assert "q2Y" in model


# =============================================================================
# CCA Integration Tests
# =============================================================================


class TestPLSCCA:
    """Test PLS with CCA option."""

    def test_pls_with_cca(self, complete_data):
        """Test PLS with CCA flag."""
        X, Y = complete_data
        model = pls(X, Y, 3, cca=True, shush=True)
        assert "Tcv" in model
        assert "Pcv" in model
        assert "Wcv" in model
        assert "Betacv" in model

    def test_pls_cca_tcv_shape(self, complete_data):
        """Test CCA covariant scores shape."""
        X, Y = complete_data
        model = pls(X, Y, 3, cca=True, shush=True)
        # Tcv has one column per Y variable
        assert model["Tcv"].shape[0] == X.shape[0]
        assert model["Tcv"].shape[1] == Y.shape[1]


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestPLSDiagnostics:
    """Test PLS diagnostic statistics."""

    def test_pls_t2_values(self, complete_data):
        """Test T² values are computed."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert model["T2"].shape[0] == X.shape[0]
        assert np.all(model["T2"] >= 0)

    def test_pls_t2_limits(self, complete_data):
        """Test T² limits are computed."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert "T2_lim95" in model
        assert "T2_lim99" in model
        assert model["T2_lim99"] > model["T2_lim95"]

    def test_pls_spex_values(self, complete_data):
        """Test SPE_X values are computed."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert model["speX"].shape[0] == X.shape[0]
        assert np.all(model["speX"] >= 0)

    def test_pls_spey_values(self, complete_data):
        """Test SPE_Y values are computed."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert model["speY"].shape[0] == Y.shape[0]
        assert np.all(model["speY"] >= 0)

    def test_pls_spe_limits(self, complete_data):
        """Test SPE limits are computed."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        assert "speX_lim95" in model
        assert "speX_lim99" in model
        assert "speY_lim95" in model
        assert "speY_lim99" in model


# =============================================================================
# Prep for NLP Tests
# =============================================================================


class TestPrepPLS4NLP:
    """Test prep_pls_4_MDbyNLP function."""

    def test_prep_returns_dict(self, complete_data):
        """Test that prep returns a dict."""
        X, Y = complete_data
        model = pls_(X, Y, 2, shush=True)
        prepped = prep_pls_4_MDbyNLP(model, X, Y)
        assert isinstance(prepped, dict)

    def test_prep_adds_pyomo_keys(self, complete_data):
        """Test that prep adds Pyomo-formatted keys."""
        X, Y = complete_data
        model = pls_(X, Y, 2, shush=True)
        prepped = prep_pls_4_MDbyNLP(model, X, Y)
        pyomo_keys = [
            "pyo_A", "pyo_N", "pyo_O", "pyo_M",
            "pyo_P_init", "pyo_T_init", "pyo_Q_init",
            "pyo_X", "pyo_Y", "pyo_psi", "pyo_theta",
        ]
        for key in pyomo_keys:
            assert key in prepped, f"Missing Pyomo key: {key}"


# =============================================================================
# Edge Cases
# =============================================================================


class TestPLSEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_pls_single_component(self, complete_data):
        """Test PLS with single component."""
        X, Y = complete_data
        model = pls(X, Y, 1, shush=True)
        assert model["T"].shape[1] == 1
        assert model["P"].shape[1] == 1
        assert model["Q"].shape[1] == 1

    def test_pls_many_components(self, complete_data):
        """Test PLS with many components."""
        X, Y = complete_data
        # Don't exceed min(n_samples, n_features)
        n_components = min(X.shape) - 1
        model = pls(X, Y, n_components, shush=True)
        assert model["T"].shape[1] == n_components

    def test_pls_wide_x(self):
        """Test PLS with more X features than samples."""
        np.random.seed(42)
        X = np.random.randn(20, 50)  # Wide matrix
        Y = np.random.randn(20, 2)
        model = pls(X, Y, 2, shush=True)
        assert "T" in model

    def test_pls_many_y_variables(self):
        """Test PLS with many Y variables."""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        Y = np.random.randn(50, 15)  # Many Y variables
        model = pls(X, Y, 3, shush=True)
        assert model["Q"].shape[0] == 15


# =============================================================================
# Output Consistency Tests
# =============================================================================


class TestPLSConsistency:
    """Test output consistency and mathematical properties."""

    def test_pls_reproducible(self, complete_data):
        """Test that PLS produces reproducible results."""
        X, Y = complete_data
        model1 = pls(X, Y, 3, shush=True)
        model2 = pls(X, Y, 3, shush=True)
        np.testing.assert_array_almost_equal(model1["T"], model2["T"])

    def test_pls_scores_orthogonal(self, complete_data):
        """Test that scores are approximately orthogonal."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        T = model["T"]
        # T'T should be approximately diagonal
        TtT = T.T @ T
        # Off-diagonal elements should be small relative to diagonal
        diag = np.diag(TtT)
        off_diag = TtT - np.diag(diag)
        # Check off-diagonal is relatively small
        assert np.max(np.abs(off_diag)) < 0.1 * np.max(diag)

    def test_pls_loadings_unit_length(self, complete_data):
        """Test that weight vectors are approximately unit length."""
        X, Y = complete_data
        model = pls(X, Y, 3, shush=True)
        W = model["W"]
        for i in range(W.shape[1]):
            norm = np.linalg.norm(W[:, i])
            assert np.isclose(norm, 1.0, atol=1e-6)
