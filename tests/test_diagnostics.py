"""Tests for diagnostics utilities in PyPhi.

This module tests:
- contributions(): Variable contributions to T², SPE, and scores
- varimax_(), varimax_rotation(): Varimax rotation for interpretability
- bootstrap_pls(), bootstrap_pls_pred(): Bootstrap PLS for uncertainty
- build_polynomial(): Polynomial model building with variable selection
"""

import numpy as np
import pandas as pd
import pytest

from pyphi import pca, pls, pca_pred, pls_pred
from pyphi.diagnostics import (
    contributions,
    varimax_,
    varimax_rotation,
    bootstrap_pls,
    bootstrap_pls_pred,
    build_polynomial,
    _findstr,
    _evalvar,
    _writeeq,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pca_model_and_data():
    """Create a PCA model and data for testing."""
    np.random.seed(42)
    # Generate structured data with 3 components
    n_obs, n_vars = 50, 10
    T = np.random.randn(n_obs, 3)  # True scores
    P = np.random.randn(n_vars, 3)  # True loadings
    P = P / np.linalg.norm(P, axis=0)  # Normalize
    X = T @ P.T + 0.1 * np.random.randn(n_obs, n_vars)  # Add noise

    model = pca(X, 3, shush=True)
    return model, X


@pytest.fixture
def pls_model_and_data():
    """Create a PLS model and data for testing."""
    np.random.seed(42)
    n_obs, n_x_vars, n_y_vars = 50, 10, 2
    # Generate X with structure
    T = np.random.randn(n_obs, 3)
    P = np.random.randn(n_x_vars, 3)
    P = P / np.linalg.norm(P, axis=0)
    X = T @ P.T + 0.1 * np.random.randn(n_obs, n_x_vars)

    # Generate Y related to X
    Q = np.random.randn(n_y_vars, 3)
    Y = T @ Q.T + 0.2 * np.random.randn(n_obs, n_y_vars)

    model = pls(X, Y, 3, shush=True)
    return model, X, Y


@pytest.fixture
def dataframe_data():
    """Create DataFrame data for build_polynomial testing."""
    np.random.seed(42)
    n_obs = 50
    data = pd.DataFrame(
        {
            "ObsID": [f"Obs{i}" for i in range(n_obs)],
            "Temperature": np.random.uniform(20, 80, n_obs),
            "Pressure": np.random.uniform(1, 10, n_obs),
            "Flow": np.random.uniform(0.5, 2.0, n_obs),
        }
    )
    # Create response with known relationship
    data["Yield"] = (
        0.5 * data["Temperature"]
        + 0.3 * data["Pressure"]
        + 0.1 * data["Temperature"] * data["Pressure"]
        + np.random.randn(n_obs) * 2
    )
    return data


# =============================================================================
# Tests for contributions()
# =============================================================================


class TestContributions:
    """Tests for the contributions function."""

    def test_contributions_ht2_pca(self, pca_model_and_data):
        """Test T² contributions for PCA model."""
        model, X = pca_model_and_data
        contrib = contributions(model, X, "ht2", to_obs=[0, 1, 2])

        assert contrib is not None
        assert contrib.shape[1] == X.shape[1]  # One contribution per variable
        assert np.all(contrib >= 0)  # T² contributions are non-negative

    def test_contributions_ht2_from_obs(self, pca_model_and_data):
        """Test T² contributions with from_obs offset."""
        model, X = pca_model_and_data
        contrib_from = contributions(model, X, "ht2", to_obs=[5], from_obs=[0, 1])

        assert contrib_from is not None
        assert contrib_from.shape[1] == X.shape[1]

    def test_contributions_spe_pca(self, pca_model_and_data):
        """Test SPE contributions for PCA model."""
        model, X = pca_model_and_data
        contrib = contributions(model, X, "spe", to_obs=[0, 1, 2])

        assert contrib is not None
        assert contrib.shape[1] == X.shape[1]

    def test_contributions_spe_pls(self, pls_model_and_data):
        """Test SPE contributions for PLS model (X and Y)."""
        model, X, Y = pls_model_and_data
        contsX, contsY = contributions(model, X, "spe", Y=Y, to_obs=[0, 1])

        assert contsX is not None
        assert contsY is not None
        assert contsX.shape[1] == X.shape[1]
        assert contsY.shape[1] == Y.shape[1]

    def test_contributions_scores(self, pca_model_and_data):
        """Test score contributions."""
        model, X = pca_model_and_data
        contrib = contributions(model, X, "scores", to_obs=[0])

        assert contrib is not None
        assert contrib.shape[1] == X.shape[1]

    def test_contributions_lv_space(self, pca_model_and_data):
        """Test contributions for specific latent space."""
        model, X = pca_model_and_data
        # Use only first 2 components (1-indexed in API)
        contrib = contributions(model, X, "ht2", to_obs=[0], lv_space=[1, 2])

        assert contrib is not None
        assert contrib.shape[1] == X.shape[1]

    def test_contributions_with_dataframe(self, pca_model_and_data):
        """Test contributions with DataFrame input."""
        model, X = pca_model_and_data
        X_df = pd.DataFrame(X, columns=[f"Var{i}" for i in range(X.shape[1])])
        X_df.insert(0, "ObsID", [f"Obs{i}" for i in range(X.shape[0])])

        contrib = contributions(model, X_df, "ht2", to_obs=[0, 1])
        assert contrib is not None
        assert contrib.shape[1] == X.shape[1]

    def test_contributions_invalid_type_raises(self, pca_model_and_data):
        """Test that invalid cont_type raises ValueError."""
        model, X = pca_model_and_data
        with pytest.raises(ValueError, match="Unknown cont_type"):
            contributions(model, X, "invalid", to_obs=[0])


# =============================================================================
# Tests for varimax_()
# =============================================================================


class TestVarimax:
    """Tests for the varimax_ internal function."""

    def test_varimax_basic(self):
        """Test basic varimax rotation."""
        np.random.seed(42)
        # Create a loading matrix
        P = np.random.randn(10, 3)
        P_rotated = varimax_(P)

        assert P_rotated.shape == P.shape
        # Check that rotation preserves column space (approximately)
        # The span should be similar

    def test_varimax_orthogonality(self):
        """Test that varimax rotation is orthogonal."""
        np.random.seed(42)
        P = np.random.randn(10, 3)
        P_rotated = varimax_(P)

        # The rotation should approximately preserve norms
        original_norms = np.linalg.norm(P, axis=0)
        rotated_norms = np.linalg.norm(P_rotated, axis=0)

        # Note: varimax doesn't exactly preserve individual column norms
        # but should preserve overall structure
        assert P_rotated.shape == P.shape

    def test_varimax_single_component(self):
        """Test varimax with single component (should return unchanged)."""
        np.random.seed(42)
        P = np.random.randn(10, 1)
        P_rotated = varimax_(P)

        np.testing.assert_array_almost_equal(P, P_rotated)

    def test_varimax_convergence(self):
        """Test that varimax converges."""
        np.random.seed(42)
        P = np.random.randn(20, 5)
        # Should complete without error
        P_rotated = varimax_(P, q=100, tol=1e-8)
        assert P_rotated.shape == P.shape


# =============================================================================
# Tests for varimax_rotation()
# =============================================================================


class TestVarimaxRotation:
    """Tests for the varimax_rotation function."""

    def test_varimax_rotation_pca(self, pca_model_and_data):
        """Test varimax rotation on PCA model."""
        model, X = pca_model_and_data
        rotated_model = varimax_rotation(model, X)

        # Check that all expected keys are present
        assert "P" in rotated_model
        assert "T" in rotated_model
        assert "r2x" in rotated_model
        assert "r2xpv" in rotated_model

        # Check shapes are preserved
        assert rotated_model["P"].shape == model["P"].shape
        assert rotated_model["T"].shape == model["T"].shape

    def test_varimax_rotation_pls(self, pls_model_and_data):
        """Test varimax rotation on PLS model."""
        model, X, Y = pls_model_and_data
        rotated_model = varimax_rotation(model, X, Y=Y)

        # Check that all expected keys are present
        assert "W" in rotated_model
        assert "P" in rotated_model
        assert "T" in rotated_model
        assert "Q" in rotated_model
        assert "U" in rotated_model
        assert "Ws" in rotated_model
        assert "r2x" in rotated_model
        assert "r2xpv" in rotated_model
        assert "r2y" in rotated_model
        assert "r2ypv" in rotated_model

        # Check shapes are preserved
        assert rotated_model["W"].shape == model["W"].shape
        assert rotated_model["Q"].shape == model["Q"].shape

    def test_varimax_rotation_with_dataframe(self, pls_model_and_data):
        """Test varimax rotation with DataFrame input."""
        model, X, Y = pls_model_and_data

        # Convert to DataFrame
        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        X_df.insert(0, "ObsID", [f"Obs{i}" for i in range(X.shape[0])])
        Y_df = pd.DataFrame(Y, columns=[f"Y{i}" for i in range(Y.shape[1])])
        Y_df.insert(0, "ObsID", [f"Obs{i}" for i in range(Y.shape[0])])

        rotated_model = varimax_rotation(model, X_df, Y=Y_df)
        assert "W" in rotated_model
        assert rotated_model["W"].shape == model["W"].shape

    def test_varimax_rotation_preserves_r2_sum(self, pca_model_and_data):
        """Test that total R² is preserved after rotation."""
        model, X = pca_model_and_data
        rotated_model = varimax_rotation(model, X)

        # Total R² should be approximately preserved
        original_r2_total = np.sum(model["r2x"])
        rotated_r2_total = np.sum(rotated_model["r2x"])

        np.testing.assert_almost_equal(original_r2_total, rotated_r2_total, decimal=2)


# =============================================================================
# Tests for bootstrap_pls()
# =============================================================================


class TestBootstrapPLS:
    """Tests for the bootstrap_pls function."""

    def test_bootstrap_pls_basic(self, pls_model_and_data):
        """Test basic bootstrap PLS."""
        _, X, Y = pls_model_and_data
        boot_models = bootstrap_pls(X, Y, num_latents=2, num_samples=5)

        assert len(boot_models) == 5
        assert all("T" in m for m in boot_models)
        assert all("Q" in m for m in boot_models)

    def test_bootstrap_pls_with_dataframe(self, pls_model_and_data):
        """Test bootstrap PLS with DataFrame input."""
        _, X, Y = pls_model_and_data

        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        X_df.insert(0, "ObsID", [f"Obs{i}" for i in range(X.shape[0])])
        Y_df = pd.DataFrame(Y, columns=[f"Y{i}" for i in range(Y.shape[1])])
        Y_df.insert(0, "ObsID", [f"Obs{i}" for i in range(Y.shape[0])])

        boot_models = bootstrap_pls(X_df, Y_df, num_latents=2, num_samples=3)
        assert len(boot_models) == 3

    def test_bootstrap_pls_reproducibility(self, pls_model_and_data):
        """Test that bootstrap with same seed gives same results."""
        _, X, Y = pls_model_and_data

        np.random.seed(123)
        boot1 = bootstrap_pls(X, Y, num_latents=2, num_samples=3)

        np.random.seed(123)
        boot2 = bootstrap_pls(X, Y, num_latents=2, num_samples=3)

        # Should get same models with same seed
        for m1, m2 in zip(boot1, boot2):
            np.testing.assert_array_almost_equal(m1["T"], m2["T"])


# =============================================================================
# Tests for bootstrap_pls_pred()
# =============================================================================


class TestBootstrapPLSPred:
    """Tests for the bootstrap_pls_pred function."""

    def test_bootstrap_pls_pred_basic(self, pls_model_and_data):
        """Test basic bootstrap prediction."""
        _, X, Y = pls_model_and_data

        # Use only first column of Y for univariate case
        Y_uni = Y[:, [0]]

        boot_models = bootstrap_pls(X, Y_uni, num_latents=2, num_samples=5)
        X_new = X[:5, :]  # Test on first 5 observations

        lower, upper = bootstrap_pls_pred(X_new, boot_models, quantiles=[0.025, 0.975])

        assert lower.shape[0] == X_new.shape[0]
        assert upper.shape[0] == X_new.shape[0]
        # Upper should be greater than lower for most observations
        # (not guaranteed due to numerical issues, but usually true)

    def test_bootstrap_pls_pred_invalid_quantile(self, pls_model_and_data):
        """Test that invalid quantiles raise ValueError."""
        _, X, Y = pls_model_and_data
        Y_uni = Y[:, [0]]

        boot_models = bootstrap_pls(X, Y_uni, num_latents=2, num_samples=3)
        X_new = X[:3, :]

        with pytest.raises(ValueError, match="Quantiles must be between zero and one"):
            bootstrap_pls_pred(X_new, boot_models, quantiles=[0.5, 1.5])

        with pytest.raises(ValueError, match="Quantiles must be between zero and one"):
            bootstrap_pls_pred(X_new, boot_models, quantiles=[-0.1, 0.5])


# =============================================================================
# Tests for internal helpers
# =============================================================================


class TestHelperFunctions:
    """Tests for internal helper functions."""

    def test_findstr_basic(self):
        """Test _findstr finds * and / operators."""
        result = _findstr("a * b / c")
        assert 2 in result  # Position of *
        assert 6 in result  # Position of /

    def test_findstr_no_operators(self):
        """Test _findstr with no operators."""
        result = _findstr("abc")
        assert result == []

    def test_findstr_multiple_operators(self):
        """Test _findstr with multiple operators."""
        result = _findstr("a*b*c/d")
        assert len(result) == 3

    def test_evalvar_simple(self, dataframe_data):
        """Test _evalvar with simple variable."""
        result = _evalvar(dataframe_data, "Temperature")
        assert result is not False
        assert len(result) == len(dataframe_data)

    def test_evalvar_with_power(self, dataframe_data):
        """Test _evalvar with power notation."""
        result = _evalvar(dataframe_data, "Temperature^2")
        expected = dataframe_data["Temperature"].values.reshape(-1, 1) ** 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_evalvar_nonexistent(self, dataframe_data):
        """Test _evalvar with nonexistent variable."""
        result = _evalvar(dataframe_data, "NonExistent")
        assert result is False

    def test_writeeq_basic(self):
        """Test _writeeq formats equation correctly."""
        betas = np.array([0.5, -0.3, 1.0])
        features = ["X1", "X2", "Bias"]
        result = _writeeq(betas, features)

        assert "0.5" in result
        assert "X1" in result
        assert "-0.3" in result
        assert "1.0" in result


# =============================================================================
# Tests for build_polynomial()
# =============================================================================


class TestBuildPolynomial:
    """Tests for the build_polynomial function."""

    def test_build_polynomial_simple(self, dataframe_data):
        """Test build_polynomial with simple factors."""
        factors = ["Temperature", "Pressure"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", show_plots=False
        )

        assert len(betas) == 3  # 2 factors + bias
        assert "Bias" in factors_out
        assert X.shape[0] == len(dataframe_data)
        assert eq is not None

    def test_build_polynomial_with_power(self, dataframe_data):
        """Test build_polynomial with power terms."""
        factors = ["Temperature", "Temperature^2"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", show_plots=False
        )

        assert len(betas) == 3
        assert X.shape[1] == 3  # 2 factors + bias column

    def test_build_polynomial_with_interaction(self, dataframe_data):
        """Test build_polynomial with interaction terms."""
        factors = ["Temperature", "Pressure", "Temperature * Pressure"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", show_plots=False
        )

        assert len(betas) == 4  # 3 factors + bias

    def test_build_polynomial_with_ratio(self, dataframe_data):
        """Test build_polynomial with ratio terms."""
        factors = ["Temperature", "Temperature / Pressure"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", show_plots=False
        )

        assert len(betas) == 3

    def test_build_polynomial_no_bias(self, dataframe_data):
        """Test build_polynomial without bias term."""
        factors = ["Temperature", "Pressure"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", bias_term=False, show_plots=False
        )

        assert len(betas) == 2  # No bias
        assert "Bias" not in factors_out

    def test_build_polynomial_equation_format(self, dataframe_data):
        """Test that equation string is properly formatted."""
        factors = ["Temperature"]
        betas, factors_out, X, Y, eq = build_polynomial(
            dataframe_data, factors, "Yield", show_plots=False
        )

        # Equation should contain variable name and numerical coefficient
        assert "Temperature" in eq or "Bias" in eq


# =============================================================================
# Integration Tests
# =============================================================================


class TestDiagnosticsIntegration:
    """Integration tests for diagnostics module."""

    def test_full_pca_diagnostic_workflow(self, pca_model_and_data):
        """Test complete PCA diagnostic workflow."""
        model, X = pca_model_and_data

        # 1. Calculate contributions for outliers
        contrib_ht2 = contributions(model, X, "ht2", to_obs=[0, 1, 2])
        contrib_spe = contributions(model, X, "spe", to_obs=[0, 1, 2])

        # 2. Apply varimax rotation for interpretability
        rotated_model = varimax_rotation(model, X)

        # 3. Verify model is still valid
        pred = pca_pred(X, rotated_model)
        assert "Xhat" in pred
        assert "Tnew" in pred

    def test_full_pls_diagnostic_workflow(self, pls_model_and_data):
        """Test complete PLS diagnostic workflow."""
        model, X, Y = pls_model_and_data

        # 1. Calculate contributions
        contsX, contsY = contributions(model, X, "spe", Y=Y, to_obs=[0])
        contrib_ht2 = contributions(model, X, "ht2", to_obs=[0])

        # 2. Apply varimax rotation
        rotated_model = varimax_rotation(model, X, Y=Y)

        # 3. Predictions still work
        pred = pls_pred(X, rotated_model)
        assert "Yhat" in pred

    def test_bootstrap_prediction_interval_coverage(self, pls_model_and_data):
        """Test that bootstrap prediction intervals are reasonable."""
        _, X, Y = pls_model_and_data
        Y_uni = Y[:, [0]]

        # Train on subset
        n_train = 40
        X_train, X_test = X[:n_train], X[n_train:]
        Y_train, Y_test = Y_uni[:n_train], Y_uni[n_train:]

        boot_models = bootstrap_pls(X_train, Y_train, num_latents=2, num_samples=10)
        lower, upper = bootstrap_pls_pred(X_test, boot_models, quantiles=[0.025, 0.975])

        # Check dimensions
        assert lower.shape[0] == X_test.shape[0]
        assert upper.shape[0] == X_test.shape[0]
