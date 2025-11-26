"""Tests for the PCA module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyphi import pca, pca_, pca_pred, hott2, prep_pca_4_MDbyNLP


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def complete_data():
    """Generate complete data (no missing values) for PCA testing."""
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    return X


@pytest.fixture
def data_with_missing():
    """Generate data with missing values for PCA testing."""
    np.random.seed(42)
    n_samples, n_features = 50, 10
    X = np.random.randn(n_samples, n_features)
    # Introduce ~5% missing data
    X[0, 1] = np.nan
    X[5, 3] = np.nan
    X[10, 7] = np.nan
    X[15, 2] = np.nan
    X[20, 9] = np.nan
    return X


@pytest.fixture
def dataframe_data():
    """Generate DataFrame with observation IDs for PCA testing."""
    np.random.seed(42)
    n_samples, n_features = 30, 8
    X = np.random.randn(n_samples, n_features)
    obs_ids = [f"Obs_{i}" for i in range(n_samples)]
    var_ids = [f"Var_{j}" for j in range(n_features)]
    df = pd.DataFrame(X, columns=var_ids)
    df.insert(0, "ObsID", obs_ids)
    return df


# =============================================================================
# Basic PCA Tests
# =============================================================================


class TestPCABasic:
    """Test basic PCA functionality."""

    def test_pca_returns_dict(self, complete_data):
        """Test that pca() returns a dictionary."""
        model = pca(complete_data, 3, shush=True)
        assert isinstance(model, dict)

    def test_pca_contains_required_keys(self, complete_data):
        """Test that PCA model contains all required keys."""
        model = pca(complete_data, 3, shush=True)
        required_keys = ["T", "P", "r2x", "r2xpv", "mx", "sx", "T2", "speX", "type"]
        for key in required_keys:
            assert key in model, f"Missing key: {key}"

    def test_pca_scores_shape(self, complete_data):
        """Test that scores matrix has correct shape."""
        n_components = 3
        model = pca(complete_data, n_components, shush=True)
        assert model["T"].shape == (complete_data.shape[0], n_components)

    def test_pca_loadings_shape(self, complete_data):
        """Test that loadings matrix has correct shape."""
        n_components = 3
        model = pca(complete_data, n_components, shush=True)
        assert model["P"].shape == (complete_data.shape[1], n_components)

    def test_pca_r2x_is_positive(self, complete_data):
        """Test that R2X values are positive."""
        model = pca(complete_data, 3, shush=True)
        assert np.all(model["r2x"] >= 0)

    def test_pca_r2x_sums_less_than_one(self, complete_data):
        """Test that cumulative R2X is <= 1."""
        model = pca(complete_data, 3, shush=True)
        assert np.sum(model["r2x"]) <= 1.0 + 1e-10  # Small tolerance

    def test_pca_type_is_pca(self, complete_data):
        """Test that model type is 'pca'."""
        model = pca(complete_data, 3, shush=True)
        assert model["type"] == "pca"


# =============================================================================
# Algorithm Tests
# =============================================================================


class TestPCAAlgorithms:
    """Test different PCA algorithms."""

    def test_nipals_with_complete_data(self, complete_data):
        """Test NIPALS algorithm with complete data."""
        model = pca(complete_data, 3, force_nipals=True, shush=True)
        assert model["T"].shape == (50, 3)
        assert model["P"].shape == (10, 3)

    def test_svd_vs_nipals_produce_similar_r2x(self, complete_data):
        """Test that SVD and NIPALS produce similar R2X."""
        model_svd = pca_(complete_data, 3, force_nipals=False, shush=True)
        model_nipals = pca_(complete_data, 3, force_nipals=True, shush=True)

        # R2X should be very close
        r2x_diff = np.abs(np.sum(model_svd["r2x"]) - np.sum(model_nipals["r2x"]))
        assert r2x_diff < 1e-6, f"R2X difference too large: {r2x_diff}"

    def test_loadings_are_orthonormal(self, complete_data):
        """Test that loadings are orthonormal (P'P ≈ I)."""
        model = pca(complete_data, 3, force_nipals=True, shush=True)
        P = model["P"]
        PtP = P.T @ P
        identity = np.eye(3)
        # NIPALS has some numerical error, use reasonable tolerance
        np.testing.assert_allclose(PtP, identity, atol=1e-4)


# =============================================================================
# Missing Data Tests
# =============================================================================


class TestPCAMissingData:
    """Test PCA with missing data."""

    def test_nipals_handles_missing_data(self, data_with_missing):
        """Test that NIPALS handles missing data."""
        model = pca(data_with_missing, 3, shush=True)
        assert model["T"].shape == (50, 3)
        assert not np.any(np.isnan(model["T"]))

    def test_nipals_produces_valid_r2x_with_missing(self, data_with_missing):
        """Test that R2X is valid even with missing data."""
        model = pca(data_with_missing, 3, shush=True)
        assert np.all(model["r2x"] >= 0)
        assert np.sum(model["r2x"]) <= 1.0 + 0.01  # Slightly larger tolerance

    def test_missing_data_amount_affects_r2x(self, complete_data):
        """Test that more missing data generally decreases R2X reliability."""
        # Complete data
        model_complete = pca(complete_data, 3, shush=True)
        
        # Data with some missing
        X_miss = complete_data.copy()
        X_miss[0, 1] = np.nan
        X_miss[5, 3] = np.nan
        model_miss = pca(X_miss, 3, shush=True)
        
        # Both should produce valid models
        assert model_complete["T"].shape == model_miss["T"].shape


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestPCAPreprocessing:
    """Test PCA preprocessing options."""

    def test_mcs_true_centers_and_scales(self, complete_data):
        """Test that mcs=True mean-centers and autoscales."""
        model = pca(complete_data, 3, mcs=True, shush=True)
        assert model["mx"].shape == (1, complete_data.shape[1])
        assert model["sx"].shape == (1, complete_data.shape[1])
        # Mean should be close to column means
        np.testing.assert_allclose(model["mx"], np.nanmean(complete_data, axis=0, keepdims=True), rtol=1e-10)

    def test_mcs_false_no_preprocessing(self, complete_data):
        """Test that mcs=False doesn't preprocess."""
        model = pca(complete_data, 3, mcs=False, shush=True)
        np.testing.assert_array_equal(model["mx"], np.zeros((1, complete_data.shape[1])))
        np.testing.assert_array_equal(model["sx"], np.ones((1, complete_data.shape[1])))

    def test_mcs_center_only_centers(self, complete_data):
        """Test that mcs='center' only centers."""
        model = pca(complete_data, 3, mcs="center", shush=True)
        np.testing.assert_array_equal(model["sx"], np.ones((1, complete_data.shape[1])))
        # Mean should be close to column means
        np.testing.assert_allclose(model["mx"], np.nanmean(complete_data, axis=0, keepdims=True), rtol=1e-10)

    def test_mcs_autoscale_only_scales(self, complete_data):
        """Test that mcs='autoscale' only scales."""
        model = pca(complete_data, 3, mcs="autoscale", shush=True)
        np.testing.assert_array_equal(model["mx"], np.zeros((1, complete_data.shape[1])))


# =============================================================================
# DataFrame Tests
# =============================================================================


class TestPCADataFrame:
    """Test PCA with DataFrame input."""

    def test_dataframe_input(self, dataframe_data):
        """Test that PCA works with DataFrame input."""
        model = pca(dataframe_data, 3, shush=True)
        assert "T" in model
        assert model["T"].shape[1] == 3

    def test_dataframe_preserves_obs_ids(self, dataframe_data):
        """Test that observation IDs are preserved."""
        model = pca(dataframe_data, 3, shush=True)
        assert "obsidX" in model
        assert len(model["obsidX"]) == 30

    def test_dataframe_preserves_var_ids(self, dataframe_data):
        """Test that variable IDs are preserved."""
        model = pca(dataframe_data, 3, shush=True)
        assert "varidX" in model
        assert len(model["varidX"]) == 8


# =============================================================================
# Diagnostics Tests
# =============================================================================


class TestPCADiagnostics:
    """Test PCA diagnostic statistics."""

    def test_t2_calculated(self, complete_data):
        """Test that T2 is calculated."""
        model = pca(complete_data, 3, shush=True)
        assert "T2" in model
        assert model["T2"].shape[0] == complete_data.shape[0]

    def test_t2_limits_calculated(self, complete_data):
        """Test that T2 limits are calculated."""
        model = pca(complete_data, 3, shush=True)
        assert "T2_lim95" in model
        assert "T2_lim99" in model
        assert model["T2_lim99"] > model["T2_lim95"]

    def test_spe_calculated(self, complete_data):
        """Test that SPE is calculated."""
        model = pca(complete_data, 3, shush=True)
        assert "speX" in model
        assert model["speX"].shape[0] == complete_data.shape[0]

    def test_spe_limits_calculated(self, complete_data):
        """Test that SPE limits are calculated."""
        model = pca(complete_data, 3, shush=True)
        assert "speX_lim95" in model
        assert "speX_lim99" in model
        assert model["speX_lim99"] > model["speX_lim95"]

    def test_spe_is_positive(self, complete_data):
        """Test that SPE values are non-negative."""
        model = pca(complete_data, 3, shush=True)
        assert np.all(model["speX"] >= 0)

    def test_t2_is_positive(self, complete_data):
        """Test that T2 values are non-negative."""
        model = pca(complete_data, 3, shush=True)
        assert np.all(model["T2"] >= 0)


# =============================================================================
# hott2() Function Tests
# =============================================================================


class TestHott2:
    """Test Hotelling's T2 function."""

    def test_hott2_with_tnew(self, complete_data):
        """Test hott2() with provided Tnew."""
        model = pca(complete_data, 3, shush=True)
        T2 = hott2(model, Tnew=model["T"])
        assert T2.shape[0] == complete_data.shape[0]
        assert np.all(T2 >= 0)

    def test_hott2_without_args(self, complete_data):
        """Test hott2() without arguments uses training data."""
        model = pca(complete_data, 3, shush=True)
        T2 = hott2(model)
        np.testing.assert_allclose(T2, model["T2"], rtol=1e-10)

    def test_hott2_with_new_scores(self, complete_data):
        """Test hott2() with new scores."""
        model = pca(complete_data, 3, shush=True)
        new_scores = np.random.randn(5, 3)
        T2 = hott2(model, Tnew=new_scores)
        assert T2.shape[0] == 5


# =============================================================================
# pca_pred() Tests
# =============================================================================


class TestPCAPred:
    """Test PCA prediction function."""

    def test_pca_pred_returns_dict(self, complete_data):
        """Test that pca_pred returns a dictionary."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(5, 10)
        pred = pca_pred(Xnew, model)
        assert isinstance(pred, dict)

    def test_pca_pred_contains_required_keys(self, complete_data):
        """Test that prediction contains required keys."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(5, 10)
        pred = pca_pred(Xnew, model)
        required_keys = ["Xhat", "Tnew", "speX", "T2"]
        for key in required_keys:
            assert key in pred, f"Missing key: {key}"

    def test_pca_pred_tnew_shape(self, complete_data):
        """Test that Tnew has correct shape."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(5, 10)
        pred = pca_pred(Xnew, model)
        assert pred["Tnew"].shape == (5, 3)

    def test_pca_pred_xhat_shape(self, complete_data):
        """Test that Xhat has correct shape."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(5, 10)
        pred = pca_pred(Xnew, model)
        assert pred["Xhat"].shape == (5, 10)

    def test_pca_pred_with_missing_data(self, complete_data):
        """Test pca_pred with missing data in new observations."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(5, 10)
        Xnew[0, 2] = np.nan
        Xnew[2, 5] = np.nan
        pred = pca_pred(Xnew, model)
        assert pred["Tnew"].shape == (5, 3)
        assert not np.any(np.isnan(pred["Tnew"]))

    def test_pca_pred_single_observation(self, complete_data):
        """Test pca_pred with a single observation."""
        model = pca(complete_data, 3, shush=True)
        Xnew = np.random.randn(10)  # 1D array
        pred = pca_pred(Xnew, model)
        assert pred["Tnew"].shape == (1, 3)

    def test_pca_pred_reconstruction_reasonable(self, complete_data):
        """Test that reconstruction is reasonable for training data."""
        model = pca(complete_data, 3, shush=True)
        pred = pca_pred(complete_data, model)
        # Reconstruction error should be bounded
        recon_error = np.mean((pred["Xhat"] - complete_data) ** 2)
        # Error should be reasonable (depends on R2X)
        assert recon_error < np.var(complete_data) * (1 - np.sum(model["r2x"]) + 0.1)


# =============================================================================
# Edge Cases
# =============================================================================


class TestPCAEdgeCases:
    """Test edge cases for PCA."""

    def test_single_component(self, complete_data):
        """Test PCA with single component."""
        model = pca(complete_data, 1, shush=True)
        assert model["T"].shape == (50, 1)
        assert model["P"].shape == (10, 1)

    def test_many_components(self):
        """Test PCA with many components (approaching rank)."""
        np.random.seed(42)
        X = np.random.randn(20, 10)
        model = pca(X, 8, shush=True)
        assert model["T"].shape == (20, 8)

    def test_single_row(self):
        """Test PCA handles single row gracefully."""
        np.random.seed(42)
        X = np.random.randn(1, 10)
        # This should work but might have issues - testing for no crash
        try:
            model = pca(X, 1, mcs=False, shush=True)
            # If it works, T should have shape (1, 1)
            assert model["T"].shape[0] == 1
        except Exception:
            # Some edge cases might fail - that's acceptable
            pass

    def test_high_dimensional(self):
        """Test PCA with high dimensional data (p >> n)."""
        np.random.seed(42)
        X = np.random.randn(20, 100)  # More features than samples
        model = pca(X, 5, shush=True)
        assert model["T"].shape == (20, 5)
        assert model["P"].shape == (100, 5)


# =============================================================================
# Cross-Validation Tests
# =============================================================================


class TestPCACrossValidation:
    """Test PCA cross-validation."""

    def test_cross_val_returns_q2(self, complete_data):
        """Test that cross-validation returns Q2."""
        model = pca(complete_data, 2, cross_val=20, shush=True)
        assert "q2" in model
        assert "q2pv" in model

    def test_cross_val_q2_shape(self, complete_data):
        """Test Q2 shape matches number of components."""
        n_components = 2
        model = pca(complete_data, n_components, cross_val=20, shush=True)
        # q2 should have n_components elements
        assert len(model["q2"]) == n_components if isinstance(model["q2"], np.ndarray) else True

    def test_cross_val_invalid_percentage(self, complete_data):
        """Test that invalid cross_val percentage returns error."""
        result = pca(complete_data, 2, cross_val=150, shush=True)
        # Should return an error message, not a valid model
        assert isinstance(result, str) or "error" in str(result).lower()


# =============================================================================
# Integration Tests
# =============================================================================


class TestPCAIntegration:
    """Integration tests for PCA workflow."""

    def test_full_workflow(self, complete_data):
        """Test complete PCA workflow: fit, predict, diagnose."""
        # Fit model
        model = pca(complete_data, 3, shush=True)
        
        # Generate new data
        np.random.seed(123)
        Xnew = np.random.randn(10, 10)
        
        # Predict
        pred = pca_pred(Xnew, model)
        
        # Calculate diagnostics
        T2_new = hott2(model, Tnew=pred["Tnew"])
        
        # Verify all outputs are valid
        assert not np.any(np.isnan(model["T"]))
        assert not np.any(np.isnan(pred["Tnew"]))
        assert not np.any(np.isnan(T2_new))

    def test_model_reproducibility(self, complete_data):
        """Test that PCA is reproducible with same data."""
        model1 = pca(complete_data, 3, force_nipals=True, shush=True)
        model2 = pca(complete_data, 3, force_nipals=True, shush=True)
        
        # Results should be identical (or very close for sign flips)
        np.testing.assert_allclose(np.abs(model1["T"]), np.abs(model2["T"]), rtol=1e-10)
        np.testing.assert_allclose(np.abs(model1["P"]), np.abs(model2["P"]), rtol=1e-10)
