# Refactor Changelog

This log tracks the gradual migration of functionality from the legacy
`pyphi.py` module into the new modular package.

Entries should include:

- Date of the change
- Functions duplicated or moved
- Associated tests that cover the change
- Any follow-up items (e.g., consumers still using the legacy copy)

## Entries

### 2025-11-26: Phase 6 Complete - Diagnostics & Analysis Tools

**New file created: `src/pyphi/diagnostics.py`**

| Function | Type | Description |
|----------|------|-------------|
| `contributions()` | **PUBLIC** | Calculate variable contributions to T², SPE, or scores |
| `varimax_()` | **PUBLIC** | Internal varimax rotation algorithm (exposed for advanced users) |
| `varimax_rotation()` | **PUBLIC** | Apply varimax rotation to PCA or PLS model |
| `bootstrap_pls()` | **PUBLIC** | Generate bootstrap PLS models for uncertainty quantification |
| `bootstrap_pls_pred()` | **PUBLIC** | Make predictions with quantiles using bootstrapped PLS |
| `build_polynomial()` | **PUBLIC** | Build polynomial regression with PLS-assisted variable selection |
| `_findstr()` | **INTERNAL** | Find operator positions in expression string |
| `_evalvar()` | **INTERNAL** | Evaluate variable expression from DataFrame |
| `_writeeq()` | **INTERNAL** | Format regression equation as string |

**Features implemented:**
- ✅ `contributions()`: Supports 'ht2', 'spe', and 'scores' contribution types
- ✅ `contributions()`: Handles both PCA and PLS models
- ✅ `contributions()`: Supports DataFrame input with observation IDs
- ✅ `contributions()`: from_obs/to_obs for differential contributions
- ✅ `contributions()`: lv_space for component-specific contributions
- ✅ `varimax_()`: Core varimax rotation with configurable tolerance
- ✅ `varimax_rotation()`: Complete model rotation for PCA and PLS
- ✅ `varimax_rotation()`: Recalculates R² values after rotation
- ✅ `bootstrap_pls()`: Generates ensemble of bootstrap PLS models
- ✅ `bootstrap_pls_pred()`: Gaussian error-based prediction intervals
- ✅ `build_polynomial()`: VIP-based variable selection visualization
- ✅ `build_polynomial()`: Supports powers, interactions, and ratios

**Tests verified (37 tests in `tests/test_diagnostics.py`):**
- `TestContributions`: 8 tests for contribution analysis
- `TestVarimax`: 4 tests for varimax rotation algorithm
- `TestVarimaxRotation`: 4 tests for model rotation
- `TestBootstrapPLS`: 3 tests for bootstrap PLS
- `TestBootstrapPLSPred`: 2 tests for bootstrap predictions
- `TestHelperFunctions`: 7 tests for internal helpers
- `TestBuildPolynomial`: 6 tests for polynomial model building
- `TestDiagnosticsIntegration`: 3 integration tests

**All exports available:**
```python
from pyphi import (
    contributions,
    varimax_,
    varimax_rotation,
    bootstrap_pls,
    bootstrap_pls_pred,
    build_polynomial,
)
```

**Dependencies:**
- Imports from `.utils`: n2z
- Imports from `._internal`: _Ab_btbinv
- Lazy imports from `.pca` and `.pls` to avoid circular dependencies
- Uses scipy.stats.norm and scipy.optimize.fsolve for bootstrap predictions
- Uses matplotlib.pyplot for build_polynomial visualizations

**Notes:**
- `hott2()` remains in `pca.py` (per existing architecture)
- `spe()` remains in `pls.py` (per existing architecture)
- `cca()` and `cca_multi()` remain in `pls.py` (per existing architecture)
- These functions are still exported at package level from their original modules

**Next Phase:** Phase 7 (Exports) - export_2_gproms, adapt_pls_4_pyomo, parse_materials, cat_2_matrix, etc.

---

### 2025-11-26: Phase 1 Complete - Data Cleaning & Internal Helpers (Batch 3)

**Functions added to `src/pyphi/utils.py`:**

| Function | Type | Description |
|----------|------|-------------|
| `clean_htmls()` | **KEPT CUSTOM** | Remove HTML files from current directory |
| `clean_empty_rows()` | **KEPT CUSTOM** | Remove rows with all-NaN data |
| `clean_low_variances()` | **KEPT CUSTOM** | Remove columns with low variance or too much missing data |
| `isin_ordered_col0()` | **KEPT CUSTOM** | Filter and reorder DataFrame by first column values |
| `reconcile_rows()` | **KEPT CUSTOM** | Align rows across multiple DataFrames |
| `reconcile_rows_to_columns()` | **KEPT CUSTOM** | Align rows of one list with columns of another |

**Functions added to `src/pyphi/_internal.py`:**

| Function | Type | Description |
|----------|------|-------------|
| `scores_conf_int_calc()` | **KEPT CUSTOM** | Calculate bivariate score confidence ellipse points |
| `_Ab_btbinv()` | **KEPT CUSTOM** | Matrix projection with missing data handling |
| `np2D2pyomo()` | **KEPT CUSTOM** | Convert 2D NumPy array to Pyomo dictionary format |
| `np1D2pyomo()` | **KEPT CUSTOM** | Convert 1D NumPy array to Pyomo dictionary format |

**All Phase 1 functions now in modular package:**

```python
from pyphi import (
    # Statistical (Batch 1)
    mean, std, meancenterscale, f95, f99,
    # Confidence intervals (Batch 2)
    spe_ci, single_score_conf_int,
    # NaN utilities (Batch 2)
    n2z, z2n,
    # Array utilities (Batch 2)
    find, unique,
    # Data cleaning (Batch 3)
    clean_htmls, clean_empty_rows, clean_low_variances,
    # DataFrame reconciliation (Batch 3)
    isin_ordered_col0, reconcile_rows, reconcile_rows_to_columns,
    # Internal helpers (Batch 3)
    scores_conf_int_calc, np2D2pyomo, np1D2pyomo,
)
```

**Phase 1 Summary:**
- ✅ 22 functions migrated to modular package
- ✅ ~200 lines of code reduced via library swaps
- ✅ All statistical functions use exact scipy.stats calculations
- ✅ Full test coverage with manual verification
- ✅ Legacy implementations preserved in `pyphi_legacy.py`

**Next Phase:** Phase 6 (Diagnostics) - hott2, spe, contributions, varimax_rotation, bootstrap_pls, etc.

---

### 2025-11-26: Phase 5 - Advanced PLS Module

**New file created: `src/pyphi/advanced_pls.py`**

| Function | Type | Description |
|----------|------|-------------|
| `lwpls()` | **PUBLIC** | Locally Weighted PLS prediction |
| `mbpls()` | **PUBLIC** | Multi-block PLS model building |
| `lpls()` | **PUBLIC** | Linear PLS (bilinear model) for materials/blending |
| `lpls_pred()` | **PUBLIC** | LPLS prediction function |
| `jrpls()` | **PUBLIC** | Joint Range PLS for multi-material blending |
| `jrpls_pred()` | **PUBLIC** | JRPLS prediction function |
| `tpls()` | **PUBLIC** | Tensor/Three-way PLS with process conditions |
| `tpls_pred()` | **PUBLIC** | TPLS prediction function |

**Features implemented:**
- ✅ LWPLS: Locally weighted prediction using VIP-based distance weighting
- ✅ MBPLS: Multi-block PLS with block-specific loadings and super weights
- ✅ LPLS: Bilinear model for material properties × blending ratios → quality
- ✅ JRPLS: Joint range PLS for multiple materials with separate properties
- ✅ TPLS: Three-way PLS including process conditions (materials + process → quality)
- ✅ All prediction functions support both list and dictionary input formats
- ✅ Full diagnostic outputs (T², SPE, R² values)
- ✅ DataFrame support with observation/variable IDs

**Tests verified (22 tests in `tests/test_advanced_pls.py`):**
- `TestLWPLS`: 3 tests for locally weighted PLS
- `TestMBPLS`: 4 tests for multi-block PLS
- `TestLPLS`: 5 tests for linear PLS and predictions
- `TestJRPLS`: 4 tests for joint range PLS and predictions
- `TestTPLS`: 4 tests for tensor PLS and predictions
- `TestAdvancedPLSIntegration`: 2 integration tests

**All exports available:**
```python
from pyphi import (
    lwpls,
    mbpls,
    lpls, lpls_pred,
    jrpls, jrpls_pred,
    tpls, tpls_pred,
)
```

**References:**
- LWPLS: International Journal of Pharmaceutics 421 (2011) 269-274
- MBPLS: Westerhuis, J. Chemometrics, 12, 301-321 (1998)
- LPLS: Muteki et al. Chemom. Intell. Lab. Syst. 85 (2007) 186-194
- JRPLS/TPLS: Garcia-Munoz Chemom. Intel. Lab. Syst., 133, pp.49-62

---

### 2025-11-26: Phase 4 - PLS Module

**New file created: `src/pyphi/pls.py`**

| Function | Type | Description |
|----------|------|-------------|
| `pls()` | **PUBLIC** | Main PLS wrapper with cross-validation support |
| `pls_()` | **PUBLIC** | Core PLS algorithm (SVD, NIPALS, NLP) |
| `pls_pred()` | **PUBLIC** | Predict Y values and project new X data |
| `spe()` | **PUBLIC** | Calculate Squared Prediction Error |
| `prep_pls_4_MDbyNLP()` | **PUBLIC** | Prepare PLS for NLP optimization |
| `cca()` | **PUBLIC** | Canonical Correlation Analysis |
| `cca_multi()` | **PUBLIC** | Multi-component CCA |
| `_pls_svd()` | **INTERNAL** | SVD algorithm for complete data |
| `_pls_nipals()` | **INTERNAL** | NIPALS algorithm (handles missing data) |
| `_pls_nlp()` | **INTERNAL** | NLP optimization via Pyomo |
| `_pls_cross_validate()` | **INTERNAL** | Cross-validation implementation |
| `_pls_cca()` | **INTERNAL** | CCA for OPLS-like covariant calculation |
| `_add_pls_diagnostics()` | **INTERNAL** | Add T², SPE to model object |

**Features implemented:**
- ✅ SVD algorithm for complete data
- ✅ NIPALS algorithm for missing data handling
- ✅ NLP optimization via Pyomo (optional, for advanced missing data)
- ✅ Cross-validation (element-wise and leave-one-out)
- ✅ Multiple preprocessing options for both X and Y
- ✅ Hotelling's T² and SPE diagnostics with confidence limits (for both X and Y)
- ✅ DataFrame support with observation/variable IDs
- ✅ Prediction function with missing data support (p2mp algorithm)
- ✅ CCA option for OPLS-like covariant scores/loadings
- ✅ Multi-block data support in predictions

**Tests verified (57 tests total):**
- `pls()` with complete data
- `pls()` with NIPALS and SVD algorithms
- `pls()` with missing data
- `pls_pred()` for new data projection
- `pls_pred()` with missing data
- `spe()` for SPE calculation (X and Y)
- `cca()` and `cca_multi()` for canonical correlation
- Cross-validation (element-wise and leave-one-out)
- CCA option for OPLS-like behavior
- All preprocessing options (mcsX/mcsY=True/False/'center'/'autoscale')
- DataFrame input support
- Diagnostic statistics (T², SPE, limits)
- Edge cases (univariate Y, wide X, many Y variables)

**All exports available:**
```python
from pyphi import pls, pls_, pls_pred, spe, prep_pls_4_MDbyNLP, cca, cca_multi
```

**Note:** NLP algorithm requires Pyomo and IPOPT/GAMS. Falls back gracefully if unavailable.

---

### 2025-11-26: Phase 3 - PCA Module

**New file created: `src/pyphi/pca.py`**

| Function | Type | Description |
|----------|------|-------------|
| `pca()` | **PUBLIC** | Main PCA wrapper with cross-validation support |
| `pca_()` | **PUBLIC** | Core PCA algorithm (SVD, NIPALS, NLP) |
| `pca_pred()` | **PUBLIC** | Project new data onto existing PCA model |
| `hott2()` | **PUBLIC** | Calculate Hotelling's T² statistic |
| `prep_pca_4_MDbyNLP()` | **PUBLIC** | Prepare PCA for NLP optimization |
| `_pca_svd()` | **INTERNAL** | SVD algorithm for complete data |
| `_pca_nipals()` | **INTERNAL** | NIPALS algorithm (handles missing data) |
| `_pca_nlp()` | **INTERNAL** | NLP optimization via Pyomo |
| `_pca_cross_validate()` | **INTERNAL** | Cross-validation implementation |
| `_add_pca_diagnostics()` | **INTERNAL** | Add T², SPE to model object |

**Features implemented:**
- ✅ SVD algorithm for complete data (auto-selected for wide/tall matrices)
- ✅ NIPALS algorithm for missing data handling
- ✅ NLP optimization via Pyomo (optional, for advanced missing data)
- ✅ Cross-validation for model selection
- ✅ Multiple preprocessing options (center, autoscale, both, none)
- ✅ Hotelling's T² and SPE diagnostics with confidence limits
- ✅ DataFrame support with observation/variable IDs
- ✅ Prediction function with missing data support (p2mp algorithm)

**Tests verified:**
- `pca()` with complete data
- `pca()` with NIPALS (force_nipals=True)
- `pca()` with missing data
- `pca_pred()` for new data projection
- `pca_pred()` with missing data
- `hott2()` for T² calculation
- All preprocessing options (mcs=True/False/'center'/'autoscale')

**All exports available:**
```python
from pyphi import pca, pca_, pca_pred, hott2, prep_pca_4_MDbyNLP
```

**Note:** NLP algorithm requires Pyomo and IPOPT/GAMS. Falls back gracefully if unavailable

---

### 2025-11-26: Phase 1 Batch 2 - Additional Utility Functions

**Functions added to `src/pyphi/utils.py`:**

| Function | Type | Implementation |
|----------|------|----------------|
| `n2z()` | **KEPT CUSTOM** | Simple NaN→zero conversion with map tracking |
| `z2n()` | **KEPT CUSTOM** | Restores NaN from map (inverse of n2z) |
| `spe_ci()` | **LIBRARY SWAP** | Uses `scipy.stats.chi2.ppf()` instead of lookup table (~40 lines removed) |
| `single_score_conf_int()` | **LIBRARY SWAP** | Uses `scipy.stats.t.ppf()` instead of lookup table (~35 lines removed) |
| `find()` | **KEPT CUSTOM** | Simple list comprehension (no benefit from np.where for callable) |
| `unique()` | **LIBRARY SWAP** | Uses `pd.unique().tolist()` (~15 lines simplified) |

**Tests added in `tests/test_utils.py`:**
- `test_n2z_converts_nan_to_zero` - NaN conversion
- `test_n2z_no_nan` - handles no-NaN case
- `test_z2n_restores_nan` - NaN restoration
- `test_n2z_z2n_roundtrip` - verifies roundtrip consistency
- `test_spe_ci_returns_positive_limits` - basic limit check
- `test_spe_ci_near_zero_spe` - edge case handling
- `test_spe_ci_uses_chi2` - verifies scipy.stats.chi2 calculation
- `test_single_score_conf_int_returns_positive` - basic check
- `test_single_score_conf_int_uses_t_dist` - verifies scipy.stats.t calculation
- `test_single_score_conf_int_sample_size_effect` - statistical behavior
- `test_find_basic`, `test_find_no_matches`, `test_find_all_match`, `test_find_with_equality`
- `test_unique_preserves_order`, `test_unique_single_value`, `test_unique_all_different`, `test_unique_with_strings`

**Benefits:**
- ~90 lines of lookup table code replaced with exact scipy.stats calculations
- `spe_ci()` now uses exact chi-squared quantiles (was interpolated from 38-row table)
- `single_score_conf_int()` now uses exact t-distribution quantiles (was interpolated from 35-row table)
- All functions exported at package level: `from pyphi import spe_ci, find, unique, ...`

**Legacy status:** Original implementations remain in `pyphi_legacy.py`.

---

### 2025-11-26: Library Swaps for Phase 1 Utilities (Batch 1)

**Functions updated in `src/pyphi/utils.py`:**

| Function | Change | Before | After |
|----------|--------|--------|-------|
| `mean()` | **SWAPPED** | 12-line custom implementation | `np.nanmean(X, axis=0, keepdims=True)` |
| `std()` | **SWAPPED** | 22-line custom implementation | `np.nanstd(X, axis=0, keepdims=True, ddof=1)` |
| `meancenterscale()` | **SIMPLIFIED** | Called custom `mean()`/`std()` with `np.tile()` | Direct numpy nan-functions, cleaner logic |
| `f95()` | **NEW** | N/A (not yet in module) | `scipy.stats.f.ppf(0.95, dfn, dfd)` |
| `f99()` | **NEW** | N/A (not yet in module) | `scipy.stats.f.ppf(0.99, dfn, dfd)` |

**Tests added/updated in `tests/test_utils.py`:**
- `test_mean_ignores_nan` - verifies NaN handling
- `test_mean_without_nan` - verifies standard case
- `test_mean_all_nan_column` - edge case for all-NaN columns
- `test_std_matches_nanstd` - verifies ddof=1 behavior
- `test_std_without_nan` - verifies standard case
- `test_meancenterscale_with_nan` - NaN handling in scaling
- `test_f95_matches_scipy` - verifies exact scipy match
- `test_f99_matches_scipy` - verifies exact scipy match
- `test_f95_known_values` - sanity check against known values
- `test_f99_known_values` - sanity check against known values
- `test_f95_f99_relationship` - verifies f99 > f95

**Benefits:**
- ~35 lines of custom code replaced with ~10 lines using standard libraries
- Exact scipy.stats calculations instead of interpolated lookup tables (for f95/f99)
- Better edge case handling (all-NaN columns, etc.)
- No new dependencies (numpy and scipy already required)

**Legacy status:** Original implementations remain in `pyphi_legacy.py` for backward compatibility.

---

### 2025-01-15: Initial Duplication

Duplicated `mean`, `std`, and `meancenterscale` into `src/pyphi/utils.py`; added comprehensive tests in `tests/test_utils.py`; legacy implementations remain in `pyphi.py`.
