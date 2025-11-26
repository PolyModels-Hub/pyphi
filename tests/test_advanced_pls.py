"""Tests for advanced PLS models (Phase 5).

This module tests:
- LWPLS: Locally Weighted PLS
- MBPLS: Multi-block PLS
- LPLS: Linear PLS (bilinear model)
- JRPLS: Joint Range PLS
- TPLS: Tensor/Three-way PLS
"""

import numpy as np
import pandas as pd
import pytest

from pyphi import (
    lwpls,
    mbpls,
    lpls,
    lpls_pred,
    jrpls,
    jrpls_pred,
    tpls,
    tpls_pred,
    pls,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_pls_data():
    """Generate simple PLS training data with univariate Y (for LWPLS)."""
    np.random.seed(42)
    n_samples = 50
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    # Create univariate Y with some correlation to X (LWPLS is designed for univariate Y)
    Y = X[:, 0:1] + np.random.randn(n_samples, 1) * 0.1
    
    return X, Y


@pytest.fixture
def pls_model(simple_pls_data):
    """Build a PLS model for LWPLS tests."""
    X, Y = simple_pls_data
    return pls(X, Y, 3, shush=True)


@pytest.fixture
def multiblock_data():
    """Generate multi-block PLS data."""
    np.random.seed(42)
    n_samples = 30
    
    # Create observation IDs
    obs_ids = [f"Obs{i}" for i in range(n_samples)]
    
    # Create X blocks with different numbers of variables
    X1 = pd.DataFrame(
        np.random.randn(n_samples, 5),
        columns=[f"X1_v{i}" for i in range(5)]
    )
    X1.insert(0, "ObsID", obs_ids)
    
    X2 = pd.DataFrame(
        np.random.randn(n_samples, 8),
        columns=[f"X2_v{i}" for i in range(8)]
    )
    X2.insert(0, "ObsID", obs_ids)
    
    X3 = pd.DataFrame(
        np.random.randn(n_samples, 4),
        columns=[f"X3_v{i}" for i in range(4)]
    )
    X3.insert(0, "ObsID", obs_ids)
    
    # Create Y data
    Y = pd.DataFrame(
        np.random.randn(n_samples, 2),
        columns=["Y1", "Y2"]
    )
    Y.insert(0, "ObsID", obs_ids)
    
    return {"X1": X1, "X2": X2, "X3": X3}, Y


@pytest.fixture
def lpls_data():
    """Generate LPLS data (materials, blending ratios, quality)."""
    np.random.seed(42)
    n_materials = 10  # Number of material lots
    n_blends = 20     # Number of blends
    n_mat_props = 5   # Number of material properties
    n_quality = 3     # Number of quality attributes
    
    # Material properties X [n_materials x n_mat_props]
    mat_ids = [f"Mat{i}" for i in range(n_materials)]
    X = pd.DataFrame(
        np.random.randn(n_materials, n_mat_props),
        columns=[f"Prop{i}" for i in range(n_mat_props)]
    )
    X.insert(0, "MatID", mat_ids)
    
    # Blending ratios R [n_blends x n_materials]
    blend_ids = [f"Blend{i}" for i in range(n_blends)]
    R_data = np.random.rand(n_blends, n_materials)
    R_data = R_data / R_data.sum(axis=1, keepdims=True)  # Normalize to sum to 1
    R = pd.DataFrame(R_data, columns=mat_ids)
    R.insert(0, "BlendID", blend_ids)
    
    # Quality Y [n_blends x n_quality]
    Y = pd.DataFrame(
        np.random.randn(n_blends, n_quality),
        columns=[f"Quality{i}" for i in range(n_quality)]
    )
    Y.insert(0, "BlendID", blend_ids)
    
    return X, R, Y


@pytest.fixture
def jrpls_data():
    """Generate JRPLS data (multiple materials with separate properties)."""
    np.random.seed(42)
    n_blends = 20
    blend_ids = [f"Blend{i}" for i in range(n_blends)]
    
    # Material 1: 5 lots, 3 properties
    mat1_lots = [f"M1_L{i}" for i in range(5)]
    Xi_mat1 = pd.DataFrame(
        np.random.randn(5, 3),
        columns=["M1_P1", "M1_P2", "M1_P3"]
    )
    Xi_mat1.insert(0, "LotID", mat1_lots)
    
    # Material 2: 4 lots, 4 properties
    mat2_lots = [f"M2_L{i}" for i in range(4)]
    Xi_mat2 = pd.DataFrame(
        np.random.randn(4, 4),
        columns=["M2_P1", "M2_P2", "M2_P3", "M2_P4"]
    )
    Xi_mat2.insert(0, "LotID", mat2_lots)
    
    Xi = {"MAT1": Xi_mat1, "MAT2": Xi_mat2}
    
    # Blending ratios for each material
    R1_data = np.random.rand(n_blends, 5)
    R1_data = R1_data / R1_data.sum(axis=1, keepdims=True)
    Ri_mat1 = pd.DataFrame(R1_data, columns=mat1_lots)
    Ri_mat1.insert(0, "BlendID", blend_ids)
    
    R2_data = np.random.rand(n_blends, 4)
    R2_data = R2_data / R2_data.sum(axis=1, keepdims=True)
    Ri_mat2 = pd.DataFrame(R2_data, columns=mat2_lots)
    Ri_mat2.insert(0, "BlendID", blend_ids)
    
    Ri = {"MAT1": Ri_mat1, "MAT2": Ri_mat2}
    
    # Quality Y
    Y = pd.DataFrame(
        np.random.randn(n_blends, 2),
        columns=["Q1", "Q2"]
    )
    Y.insert(0, "BlendID", blend_ids)
    
    return Xi, Ri, Y


@pytest.fixture
def tpls_data(jrpls_data):
    """Generate TPLS data (JRPLS + process conditions)."""
    Xi, Ri, Y = jrpls_data
    n_blends = Y.shape[0]
    blend_ids = Y["BlendID"].tolist()
    
    # Process conditions Z [n_blends x n_process_vars]
    Z = pd.DataFrame(
        np.random.randn(n_blends, 3),
        columns=["Temp", "Pressure", "Time"]
    )
    Z.insert(0, "BlendID", blend_ids)
    
    return Xi, Ri, Z, Y


# =============================================================================
# LWPLS Tests
# =============================================================================


class TestLWPLS:
    """Tests for Locally Weighted PLS."""
    
    def test_lwpls_basic(self, simple_pls_data, pls_model):
        """Test basic LWPLS prediction."""
        X, Y = simple_pls_data
        
        # Predict for first observation
        xnew = X[0, :]
        yhat = lwpls(xnew, 10.0, pls_model, X, Y, shush=True)
        
        assert yhat is not None
        # LWPLS returns a prediction (may be scalar or array)
        yhat_flat = np.ravel(yhat)
        assert len(yhat_flat) >= 1  # At least one prediction
        assert not np.any(np.isnan(yhat))
    
    def test_lwpls_different_loc_params(self, simple_pls_data, pls_model):
        """Test LWPLS with different localization parameters."""
        X, Y = simple_pls_data
        xnew = X[0, :]
        
        yhats = []
        for loc_par in [1.0, 5.0, 10.0, 50.0, 100.0]:
            yhat = lwpls(xnew, loc_par, pls_model, X, Y, shush=True)
            yhats.append(yhat)
        
        # Different loc_params should give different predictions
        # (at least some should differ)
        predictions_differ = False
        for i in range(len(yhats) - 1):
            if not np.allclose(yhats[i], yhats[i + 1], rtol=1e-5):
                predictions_differ = True
                break
        assert predictions_differ, "LWPLS should give different results for different loc_params"
    
    def test_lwpls_with_dataframe(self, simple_pls_data, pls_model):
        """Test LWPLS with DataFrame inputs."""
        X, Y = simple_pls_data
        
        # Convert to DataFrames
        X_df = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])
        X_df.insert(0, "ObsID", [f"Obs{i}" for i in range(X.shape[0])])
        
        Y_df = pd.DataFrame(Y, columns=[f"Y{i}" for i in range(Y.shape[1])])
        Y_df.insert(0, "ObsID", [f"Obs{i}" for i in range(Y.shape[0])])
        
        xnew = X[0, :]
        yhat = lwpls(xnew, 10.0, pls_model, X_df, Y_df, shush=True)
        
        assert yhat is not None
        assert not np.any(np.isnan(yhat))


# =============================================================================
# MBPLS Tests
# =============================================================================


class TestMBPLS:
    """Tests for Multi-block PLS."""
    
    def test_mbpls_basic(self, multiblock_data):
        """Test basic MBPLS model building."""
        XMB, Y = multiblock_data
        
        mbpls_obj = mbpls(XMB, Y, 2, shush_=True)
        
        assert mbpls_obj is not None
        assert mbpls_obj["type"] == "mbpls"
        assert "T" in mbpls_obj
        assert "P" in mbpls_obj
        assert "Q" in mbpls_obj
        assert "Wsb" in mbpls_obj  # Block-specific weights
        assert "Wt" in mbpls_obj   # Super weights
    
    def test_mbpls_block_structure(self, multiblock_data):
        """Test that MBPLS preserves block structure."""
        XMB, Y = multiblock_data
        
        mbpls_obj = mbpls(XMB, Y, 2, shush_=True)
        
        # Check block names are stored
        assert "Xblocknames" in mbpls_obj
        assert len(mbpls_obj["Xblocknames"]) == 3
        
        # Check block-specific weights
        assert len(mbpls_obj["Wsb"]) == 3
    
    def test_mbpls_r2_values(self, multiblock_data):
        """Test that R2 values are reasonable."""
        XMB, Y = multiblock_data
        
        mbpls_obj = mbpls(XMB, Y, 2, shush_=True)
        
        # Check R2 values exist and are in valid range
        assert "r2pbX" in mbpls_obj
        assert "r2pbXc" in mbpls_obj
        
        # Cumulative R2 should be between 0 and 1
        r2_cumulative = mbpls_obj["r2pbXc"]
        assert np.all(r2_cumulative >= 0)
        assert np.all(r2_cumulative <= 1.1)  # Allow small numerical errors
    
    def test_mbpls_with_single_dataframe(self, multiblock_data):
        """Test MBPLS with single DataFrame (no blocks)."""
        XMB, Y = multiblock_data
        
        # Use just one block as DataFrame
        X_single = XMB["X1"]
        
        mbpls_obj = mbpls(X_single, Y, 2, shush_=True)
        
        assert mbpls_obj is not None
        assert mbpls_obj["type"] == "mbpls"


# =============================================================================
# LPLS Tests
# =============================================================================


class TestLPLS:
    """Tests for Linear PLS (bilinear model)."""
    
    def test_lpls_basic(self, lpls_data):
        """Test basic LPLS model building."""
        X, R, Y = lpls_data
        
        lpls_obj = lpls(X, R, Y, 3, shush=True)
        
        assert lpls_obj is not None
        assert lpls_obj["type"] == "lpls"
        assert "T" in lpls_obj
        assert "P" in lpls_obj
        assert "Q" in lpls_obj
        assert "Rscores" in lpls_obj
        assert "S" in lpls_obj  # Material loadings
    
    def test_lpls_r2_values(self, lpls_data):
        """Test that R2 values are computed correctly."""
        X, R, Y = lpls_data
        
        lpls_obj = lpls(X, R, Y, 3, shush=True)
        
        # Check R2 for X, R, and Y
        assert "r2x" in lpls_obj
        assert "r2r" in lpls_obj
        assert "r2y" in lpls_obj
        
        # Per-variable R2 should also exist
        assert "r2xpv" in lpls_obj
        assert "r2rpv" in lpls_obj
        assert "r2ypv" in lpls_obj
    
    def test_lpls_diagnostics(self, lpls_data):
        """Test that LPLS diagnostics are computed."""
        X, R, Y = lpls_data
        
        lpls_obj = lpls(X, R, Y, 3, shush=True)
        
        # Check diagnostics
        assert "T2" in lpls_obj
        assert "T2_lim95" in lpls_obj
        assert "T2_lim99" in lpls_obj
        assert "speX" in lpls_obj
        assert "speY" in lpls_obj
        assert "speR" in lpls_obj
    
    def test_lpls_pred_basic(self, lpls_data):
        """Test basic LPLS prediction."""
        X, R, Y = lpls_data
        
        lpls_obj = lpls(X, R, Y, 3, shush=True)
        
        # Predict for first blend
        rnew = R.values[0, 1:].astype(float)  # Skip ObsID column
        preds = lpls_pred(rnew, lpls_obj)
        
        assert preds is not None
        assert "Tnew" in preds
        assert "Yhat" in preds
        assert "speR" in preds
    
    def test_lpls_pred_with_dataframe(self, lpls_data):
        """Test LPLS prediction with DataFrame input."""
        X, R, Y = lpls_data
        
        lpls_obj = lpls(X, R, Y, 3, shush=True)
        
        # Use DataFrame row for prediction
        preds = lpls_pred(R.iloc[[0]], lpls_obj)
        
        assert preds is not None
        assert "Yhat" in preds


# =============================================================================
# JRPLS Tests
# =============================================================================


class TestJRPLS:
    """Tests for Joint Range PLS."""
    
    def test_jrpls_basic(self, jrpls_data):
        """Test basic JRPLS model building."""
        Xi, Ri, Y = jrpls_data
        
        jrpls_obj = jrpls(Xi, Ri, Y, 2, shush=True)
        
        assert jrpls_obj is not None
        assert jrpls_obj["type"] == "jrpls"
        assert "T" in jrpls_obj
        assert "Q" in jrpls_obj
        assert "materials" in jrpls_obj
        assert len(jrpls_obj["materials"]) == 2
    
    def test_jrpls_material_structure(self, jrpls_data):
        """Test that JRPLS preserves material structure."""
        Xi, Ri, Y = jrpls_data
        
        jrpls_obj = jrpls(Xi, Ri, Y, 2, shush=True)
        
        # Check material-specific outputs
        assert "P" in jrpls_obj
        assert isinstance(jrpls_obj["P"], list)
        assert len(jrpls_obj["P"]) == 2
        
        assert "Rscores" in jrpls_obj
        assert isinstance(jrpls_obj["Rscores"], list)
        assert len(jrpls_obj["Rscores"]) == 2
    
    def test_jrpls_pred_with_list(self, jrpls_data):
        """Test JRPLS prediction with list input."""
        Xi, Ri, Y = jrpls_data
        
        jrpls_obj = jrpls(Xi, Ri, Y, 2, shush=True)
        
        # Create rnew as list of arrays
        rnew = [
            Ri["MAT1"].values[0, 1:].astype(float),
            Ri["MAT2"].values[0, 1:].astype(float),
        ]
        
        preds = jrpls_pred(rnew, jrpls_obj)
        
        assert preds is not None
        assert "Tnew" in preds
        assert "Yhat" in preds
        assert "speR" in preds
        assert len(preds["speR"]) == 2  # SPE per material
    
    def test_jrpls_pred_with_dict(self, jrpls_data):
        """Test JRPLS prediction with dictionary input."""
        Xi, Ri, Y = jrpls_data
        
        jrpls_obj = jrpls(Xi, Ri, Y, 2, shush=True)
        
        # Create rnew as dictionary
        mat1_lots = Xi["MAT1"]["LotID"].tolist()
        mat2_lots = Xi["MAT2"]["LotID"].tolist()
        
        rnew = {
            "MAT1": [(mat1_lots[0], 0.5), (mat1_lots[1], 0.5)],
            "MAT2": [(mat2_lots[0], 1.0)],
        }
        
        preds = jrpls_pred(rnew, jrpls_obj)
        
        assert preds is not None
        assert "Yhat" in preds


# =============================================================================
# TPLS Tests
# =============================================================================


class TestTPLS:
    """Tests for Tensor/Three-way PLS."""
    
    def test_tpls_basic(self, tpls_data):
        """Test basic TPLS model building."""
        Xi, Ri, Z, Y = tpls_data
        
        tpls_obj = tpls(Xi, Ri, Z, Y, 2, shush=True)
        
        assert tpls_obj is not None
        assert tpls_obj["type"] == "tpls"
        assert "T" in tpls_obj
        assert "Q" in tpls_obj
        assert "W" in tpls_obj  # Process weights
        assert "Pz" in tpls_obj  # Process loadings
        assert "Wt" in tpls_obj  # Super weights
    
    def test_tpls_process_space(self, tpls_data):
        """Test that TPLS includes process space outputs."""
        Xi, Ri, Z, Y = tpls_data
        
        tpls_obj = tpls(Xi, Ri, Z, Y, 2, shush=True)
        
        # Check process-specific outputs
        assert "r2z" in tpls_obj
        assert "r2zpv" in tpls_obj
        assert "mz" in tpls_obj
        assert "sz" in tpls_obj
        assert "speZ" in tpls_obj
    
    def test_tpls_pred_with_list(self, tpls_data):
        """Test TPLS prediction with list input."""
        Xi, Ri, Z, Y = tpls_data
        
        tpls_obj = tpls(Xi, Ri, Z, Y, 2, shush=True)
        
        # Create rnew as list of arrays
        rnew = [
            Ri["MAT1"].values[0, 1:].astype(float),
            Ri["MAT2"].values[0, 1:].astype(float),
        ]
        znew = Z.values[0, 1:].astype(float)
        
        preds = tpls_pred(rnew, znew, tpls_obj)
        
        assert preds is not None
        assert "Tnew" in preds
        assert "Yhat" in preds
        assert "speR" in preds
        assert "speZ" in preds
    
    def test_tpls_pred_with_dict(self, tpls_data):
        """Test TPLS prediction with dictionary input."""
        Xi, Ri, Z, Y = tpls_data
        
        tpls_obj = tpls(Xi, Ri, Z, Y, 2, shush=True)
        
        # Create rnew as dictionary
        mat1_lots = Xi["MAT1"]["LotID"].tolist()
        mat2_lots = Xi["MAT2"]["LotID"].tolist()
        
        rnew = {
            "MAT1": [(mat1_lots[0], 0.5), (mat1_lots[1], 0.5)],
            "MAT2": [(mat2_lots[0], 1.0)],
        }
        znew = Z.values[0, 1:].astype(float)
        
        preds = tpls_pred(rnew, znew, tpls_obj)
        
        assert preds is not None
        assert "Yhat" in preds
        assert "speZ" in preds


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdvancedPLSIntegration:
    """Integration tests for advanced PLS models."""
    
    def test_all_models_have_type(self, multiblock_data, lpls_data, jrpls_data, tpls_data):
        """Test that all models have a 'type' field."""
        XMB, Y_mb = multiblock_data
        X, R, Y_lpls = lpls_data
        Xi, Ri, Y_jr = jrpls_data
        Xi_t, Ri_t, Z, Y_t = tpls_data
        
        mbpls_obj = mbpls(XMB, Y_mb, 2, shush_=True)
        lpls_obj = lpls(X, R, Y_lpls, 2, shush=True)
        jrpls_obj = jrpls(Xi, Ri, Y_jr, 2, shush=True)
        tpls_obj = tpls(Xi_t, Ri_t, Z, Y_t, 2, shush=True)
        
        assert mbpls_obj["type"] == "mbpls"
        assert lpls_obj["type"] == "lpls"
        assert jrpls_obj["type"] == "jrpls"
        assert tpls_obj["type"] == "tpls"
    
    def test_all_models_have_scores(self, multiblock_data, lpls_data, jrpls_data, tpls_data):
        """Test that all models have score matrices."""
        XMB, Y_mb = multiblock_data
        X, R, Y_lpls = lpls_data
        Xi, Ri, Y_jr = jrpls_data
        Xi_t, Ri_t, Z, Y_t = tpls_data
        
        mbpls_obj = mbpls(XMB, Y_mb, 2, shush_=True)
        lpls_obj = lpls(X, R, Y_lpls, 2, shush=True)
        jrpls_obj = jrpls(Xi, Ri, Y_jr, 2, shush=True)
        tpls_obj = tpls(Xi_t, Ri_t, Z, Y_t, 2, shush=True)
        
        for model, name in [
            (mbpls_obj, "mbpls"),
            (lpls_obj, "lpls"),
            (jrpls_obj, "jrpls"),
            (tpls_obj, "tpls"),
        ]:
            assert "T" in model, f"{name} should have T matrix"
            assert model["T"].shape[1] == 2, f"{name} should have 2 components"
