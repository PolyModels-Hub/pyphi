"""Tests for model export and format conversion functions.

These tests verify the correctness of export and conversion functions
used for integrating PyPhi models with external software.
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pyphi.exports import (
    export_2_gproms,
    adapt_pls_4_pyomo,
    conv_pls_2_eiot,
    cat_2_matrix,
    parse_materials,
)
from pyphi import pls


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_pls_model():
    """Generate a simple PLS model for testing."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    Y = np.random.randn(50, 3)
    plsobj = pls(X, Y, 3, shush=True)
    return plsobj


@pytest.fixture
def sample_pls_model_with_ids():
    """Generate a PLS model with variable IDs."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, 10),
                     columns=[f'X{i}' for i in range(10)])
    X.insert(0, 'ObsID', [f'Obs{i}' for i in range(50)])
    Y = pd.DataFrame(np.random.randn(50, 3),
                     columns=['Y0', 'Y1', 'Y2'])
    Y.insert(0, 'ObsID', [f'Obs{i}' for i in range(50)])
    plsobj = pls(X, Y, 3, shush=True)
    return plsobj


@pytest.fixture
def sample_categorical_data():
    """Generate sample categorical data."""
    df = pd.DataFrame({
        'ID': ['Obs1', 'Obs2', 'Obs3', 'Obs4', 'Obs5'],
        'Color': ['Red', 'Blue', 'Red', 'Green', 'Blue'],
        'Size': ['Large', 'Small', 'Large', 'Medium', 'Small'],
        'Type': ['A', 'B', 'A', 'A', 'B']
    })
    return df


# =============================================================================
# Test: export_2_gproms
# =============================================================================

class TestExport2Gproms:
    """Tests for gPROMS export functionality."""
    
    def test_export_creates_file(self, sample_pls_model_with_ids):
        """Test that export_2_gproms creates a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_export.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=fname)
            assert os.path.exists(fname)
    
    def test_export_file_content(self, sample_pls_model_with_ids):
        """Test that exported file contains expected sections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_export.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=fname)
            
            with open(fname, 'r') as f:
                content = f.read()
            
            # Check for key sections
            assert 'PARAMETER' in content
            assert 'VARIABLE' in content
            assert 'EQUATION' in content
            assert 'ASSIGN' in content
            assert 'X_VARS' in content
            assert 'Y_VARS' in content
    
    def test_export_includes_variable_names(self, sample_pls_model_with_ids):
        """Test that exported file includes actual variable names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_export.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=fname)
            
            with open(fname, 'r') as f:
                content = f.read()
            
            # Check that X and Y variable names appear
            for xvar in sample_pls_model_with_ids['varidX']:
                assert xvar in content
            for yvar in sample_pls_model_with_ids['varidY']:
                assert yvar in content
    
    def test_export_includes_model_parameters(self, sample_pls_model_with_ids):
        """Test that exported file includes model parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_export.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=fname)
            
            with open(fname, 'r') as f:
                content = f.read()
            
            # Check for parameter assignments
            assert 'X_MEANS' in content
            assert 'X_STD' in content
            assert 'Y_MEANS' in content
            assert 'Y_STD' in content
            assert 'Ws' in content
            assert 'P' in content
            assert 'Q' in content
            assert 'Tvar' in content
    
    def test_export_number_of_components(self, sample_pls_model_with_ids):
        """Test that exported file specifies correct number of components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test_export.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=fname)
            
            with open(fname, 'r') as f:
                content = f.read()
            
            # Should have A:=3 (3 components)
            assert 'A:=3' in content


# =============================================================================
# Test: adapt_pls_4_pyomo
# =============================================================================

class TestAdaptPLS4Pyomo:
    """Tests for Pyomo adaptation functionality."""
    
    def test_adapt_returns_dict(self, sample_pls_model):
        """Test that adapt_pls_4_pyomo returns a dictionary."""
        result = adapt_pls_4_pyomo(sample_pls_model)
        assert isinstance(result, dict)
    
    def test_adapt_adds_pyomo_fields(self, sample_pls_model):
        """Test that all expected Pyomo fields are added."""
        result = adapt_pls_4_pyomo(sample_pls_model)
        
        expected_fields = [
            'pyo_A', 'pyo_N', 'pyo_M',
            'pyo_Ws', 'pyo_Q', 'pyo_P',
            'pyo_var_t', 'pyo_mx', 'pyo_sx',
            'pyo_my', 'pyo_sy', 'speX_lim95'
        ]
        
        for field in expected_fields:
            assert field in result
    
    def test_adapt_preserves_original(self, sample_pls_model):
        """Test that adaptation preserves original model fields."""
        result = adapt_pls_4_pyomo(sample_pls_model)
        
        original_fields = ['T', 'P', 'Q', 'Ws', 'mx', 'sx', 'my', 'sy']
        for field in original_fields:
            assert field in result
    
    def test_adapt_with_integer_indices(self, sample_pls_model):
        """Test adaptation with integer indices (use_var_ids=False)."""
        result = adapt_pls_4_pyomo(sample_pls_model, use_var_ids=False)
        
        # Indices should be lists of integers
        assert isinstance(result['pyo_N'], list)
        assert isinstance(result['pyo_M'], list)
        assert all(isinstance(i, int) for i in result['pyo_N'])
        assert all(isinstance(i, int) for i in result['pyo_M'])
    
    def test_adapt_with_variable_ids(self, sample_pls_model_with_ids):
        """Test adaptation with variable IDs (use_var_ids=True)."""
        result = adapt_pls_4_pyomo(sample_pls_model_with_ids, use_var_ids=True)
        
        # Indices should be lists of strings
        assert isinstance(result['pyo_N'], list)
        assert isinstance(result['pyo_M'], list)
        assert all(isinstance(i, str) for i in result['pyo_N'])
        assert all(isinstance(i, str) for i in result['pyo_M'])
    
    def test_adapt_pyomo_dicts_are_dicts(self, sample_pls_model):
        """Test that Pyomo parameters are dictionaries."""
        result = adapt_pls_4_pyomo(sample_pls_model)
        
        assert isinstance(result['pyo_Ws'], dict)
        assert isinstance(result['pyo_Q'], dict)
        assert isinstance(result['pyo_P'], dict)
        assert isinstance(result['pyo_var_t'], dict)
    
    def test_adapt_component_indices(self, sample_pls_model):
        """Test that component indices are correct."""
        result = adapt_pls_4_pyomo(sample_pls_model)
        
        A = sample_pls_model['T'].shape[1]
        assert len(result['pyo_A']) == A
        assert result['pyo_A'] == list(range(1, A + 1))


# =============================================================================
# Test: conv_pls_2_eiot
# =============================================================================

class TestConvPLS2EIOT:
    """Tests for EIOT conversion functionality."""
    
    def test_conv_returns_dict(self, sample_pls_model):
        """Test that conv_pls_2_eiot returns a dictionary."""
        result = conv_pls_2_eiot(sample_pls_model)
        assert isinstance(result, dict)
    
    def test_conv_adds_eiot_fields(self, sample_pls_model):
        """Test that EIOT-specific fields are added."""
        result = conv_pls_2_eiot(sample_pls_model)
        
        expected_fields = [
            'indx_r', 'indx_rk_eq', 'var_t', 'S_I', 'pyo_S_I'
        ]
        
        for field in expected_fields:
            assert field in result
    
    def test_conv_includes_pyomo_fields(self, sample_pls_model):
        """Test that Pyomo fields are also included."""
        result = conv_pls_2_eiot(sample_pls_model)
        
        pyomo_fields = ['pyo_A', 'pyo_N', 'pyo_M', 'pyo_Ws', 'pyo_Q', 'pyo_P']
        for field in pyomo_fields:
            assert field in result
    
    def test_conv_default_r_length(self, sample_pls_model):
        """Test conversion with default r_length (False)."""
        result = conv_pls_2_eiot(sample_pls_model, r_length=False)
        
        # Should use all X variables as R
        N = sample_pls_model['P'].shape[0]
        assert len(result['indx_r']) == N
        assert result['indx_rk_eq'] == 0
    
    def test_conv_custom_r_length(self, sample_pls_model):
        """Test conversion with custom r_length."""
        r_len = 5
        result = conv_pls_2_eiot(sample_pls_model, r_length=r_len)
        
        # First r_len variables are R, rest are equality constraints
        assert len(result['indx_r']) == r_len
        assert isinstance(result['indx_rk_eq'], list)
        assert len(result['indx_rk_eq']) > 0
    
    def test_conv_r_length_equals_n(self, sample_pls_model):
        """Test conversion when r_length equals N."""
        N = sample_pls_model['P'].shape[0]
        result = conv_pls_2_eiot(sample_pls_model, r_length=N)
        
        assert len(result['indx_r']) == N
        assert result['indx_rk_eq'] == 0
    
    def test_conv_var_t_is_array(self, sample_pls_model):
        """Test that var_t is a NumPy array."""
        result = conv_pls_2_eiot(sample_pls_model)
        assert isinstance(result['var_t'], np.ndarray)


# =============================================================================
# Test: cat_2_matrix
# =============================================================================

class TestCat2Matrix:
    """Tests for categorical-to-matrix conversion."""
    
    def test_cat_2_matrix_returns_tuple(self, sample_categorical_data):
        """Test that cat_2_matrix returns a tuple."""
        result = cat_2_matrix(sample_categorical_data)
        assert isinstance(result, tuple)
        assert len(result) == 2
    
    def test_cat_2_matrix_xmat_is_dataframe(self, sample_categorical_data):
        """Test that Xmat is a DataFrame."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        assert isinstance(Xmat, pd.DataFrame)
    
    def test_cat_2_matrix_preserves_ids(self, sample_categorical_data):
        """Test that first column (IDs) is preserved."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        assert Xmat.columns[0] == 'ID'
        assert (Xmat.iloc[:, 0] == sample_categorical_data['ID']).all()
    
    def test_cat_2_matrix_creates_binary_columns(self, sample_categorical_data):
        """Test that categorical variables are converted to binary."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        
        # Should have columns for each category
        assert 'Red' in Xmat.columns
        assert 'Blue' in Xmat.columns
        assert 'Green' in Xmat.columns
        assert 'Large' in Xmat.columns
        assert 'Small' in Xmat.columns
        assert 'Medium' in Xmat.columns
        
        # Values should be 0 or 1
        binary_cols = Xmat.iloc[:, 1:].values
        assert np.all((binary_cols == 0) | (binary_cols == 1))
    
    def test_cat_2_matrix_multiblock_structure(self, sample_categorical_data):
        """Test multi-block structure."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        
        assert isinstance(XmatMB, dict)
        assert 'data' in XmatMB
        assert 'blknames' in XmatMB
        assert isinstance(XmatMB['data'], list)
        assert isinstance(XmatMB['blknames'], list)
    
    def test_cat_2_matrix_block_count(self, sample_categorical_data):
        """Test that block count matches number of categorical columns."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        
        # Should have 3 blocks (Color, Size, Type)
        num_categorical_cols = len(sample_categorical_data.columns) - 1  # Exclude ID
        assert len(XmatMB['data']) == num_categorical_cols
        assert len(XmatMB['blknames']) == num_categorical_cols
    
    def test_cat_2_matrix_one_hot_encoding(self):
        """Test that one-hot encoding is correct."""
        df = pd.DataFrame({
            'ID': ['Obs1', 'Obs2', 'Obs3'],
            'Cat': ['A', 'B', 'A']
        })
        Xmat, XmatMB = cat_2_matrix(df)
        
        # Obs1: A=1, B=0
        assert Xmat.loc[0, 'A'] == 1
        assert Xmat.loc[0, 'B'] == 0
        
        # Obs2: A=0, B=1
        assert Xmat.loc[1, 'A'] == 0
        assert Xmat.loc[1, 'B'] == 1
        
        # Obs3: A=1, B=0
        assert Xmat.loc[2, 'A'] == 1
        assert Xmat.loc[2, 'B'] == 0


# =============================================================================
# Test: parse_materials
# =============================================================================

class TestParseMaterials:
    """Tests for materials parsing functionality."""
    
    def test_parse_materials_with_valid_data(self):
        """Test parse_materials with valid Excel data."""
        # Create sample materials data
        data = pd.DataFrame({
            'Finished Product Lot': ['FP001', 'FP001', 'FP002', 'FP002'],
            'Material Lot': ['ML_A1', 'ML_B1', 'ML_A2', 'ML_B1'],
            'Ratio or Quantity': [0.7, 0.3, 0.6, 0.4],
            'Material': ['DrugA', 'DrugB', 'DrugA', 'DrugB']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'materials.xlsx')
            data.to_excel(fname, sheet_name='Materials', index=False)
            
            JR, materials_used = parse_materials(fname, 'Materials')
            
            # Should return valid results
            assert JR is not False
            assert materials_used is not False
            assert isinstance(JR, list)
            assert isinstance(materials_used, list)
    
    def test_parse_materials_structure(self):
        """Test structure of parsed materials."""
        data = pd.DataFrame({
            'Finished Product Lot': ['FP001', 'FP001', 'FP002', 'FP002'],
            'Material Lot': ['ML_A1', 'ML_B1', 'ML_A2', 'ML_B1'],
            'Ratio or Quantity': [0.7, 0.3, 0.6, 0.4],
            'Material': ['DrugA', 'DrugB', 'DrugA', 'DrugB']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'materials.xlsx')
            data.to_excel(fname, sheet_name='Materials', index=False)
            
            JR, materials_used = parse_materials(fname, 'Materials')
            
            # Should have one DataFrame per material
            assert len(JR) == 2  # DrugA and DrugB
            assert len(materials_used) == 2
            
            # Each should be a DataFrame
            for df in JR:
                assert isinstance(df, pd.DataFrame)
    
    def test_parse_materials_with_missing_data(self):
        """Test parse_materials with missing data (should fail gracefully)."""
        data = pd.DataFrame({
            'Finished Product Lot': ['FP001', 'FP001'],
            'Material Lot': ['ML_A1', np.nan],  # Missing material lot
            'Ratio or Quantity': [0.7, 0.3],
            'Material': ['DrugA', 'DrugB']
        })
        
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'materials.xlsx')
            data.to_excel(fname, sheet_name='Materials', index=False)
            
            JR, materials_used = parse_materials(fname, 'Materials')
            
            # Should return False for both due to missing data
            assert JR is False
            assert materials_used is False


# =============================================================================
# Integration Tests
# =============================================================================

class TestExportsIntegration:
    """Integration tests for exports module."""
    
    def test_pyomo_and_eiot_compatibility(self, sample_pls_model):
        """Test that Pyomo and EIOT conversions are compatible."""
        pyomo_result = adapt_pls_4_pyomo(sample_pls_model)
        eiot_result = conv_pls_2_eiot(sample_pls_model)
        
        # EIOT should include all Pyomo fields
        pyomo_fields = ['pyo_A', 'pyo_N', 'pyo_M', 'pyo_Ws']
        for field in pyomo_fields:
            assert field in eiot_result
    
    def test_export_then_adapt_workflow(self, sample_pls_model_with_ids):
        """Test a typical workflow: build model, export, adapt."""
        # Export to gPROMS
        with tempfile.TemporaryDirectory() as tmpdir:
            gproms_file = os.path.join(tmpdir, 'model.txt')
            export_2_gproms(sample_pls_model_with_ids, fname=gproms_file)
            assert os.path.exists(gproms_file)
        
        # Adapt for Pyomo
        pyomo_model = adapt_pls_4_pyomo(sample_pls_model_with_ids, use_var_ids=True)
        assert 'pyo_Ws' in pyomo_model
        
        # Both should succeed without errors
    
    def test_categorical_to_multiblock_workflow(self, sample_categorical_data):
        """Test converting categorical data for multi-block PLS."""
        Xmat, XmatMB = cat_2_matrix(sample_categorical_data)
        
        # Multi-block structure should be ready for mbpls
        assert 'data' in XmatMB
        assert 'blknames' in XmatMB
        
        # Each block should have the ID column
        for df in XmatMB['data']:
            assert df.columns[0] == 'ID'
