# PyPhi Refactor Plan - Refined

## Overview

This document outlines the refined plan to refactor the monolithic `pyphi.py` file into a modern, testable, and maintainable Python package doing a **Full Package Restructuring**.

This approach moves the entire codebase into a proper package structure while maintaining complete backward compatibility with existing examples and user code.

---

## 📦 Target Package Structure

```
pyphi/                                # Project root
├── src/
│   └── pyphi/                        # Main package
│       ├── __init__.py              # Public API + Pyomo configuration
│       ├── _internal.py             # Private helper functions
│       ├── utils.py                 # Utility functions
│       ├── spectra.py               # Spectral preprocessing
│       ├── pca.py                   # PCA models
│       ├── pls.py                   # PLS models  
│       ├── advanced_pls.py          # Advanced model structures (LPLS, JRPLS, TPLS, MBPLS, LWPLS)
│       ├── diagnostics.py           # Diagnostics & analysis tools
│       └── exports.py               # I/O & model export
│
├── tests/                           # Test suite (create early!)
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration & fixtures
│   ├── test_utils.py
│   ├── test_spectra.py
│   ├── test_pca.py
│   ├── test_pls.py
│   ├── test_advanced_pls.py
│   ├── test_diagnostics.py
│   └── test_exports.py
│
├── notes/                            # Documentation & refactor tracking
│   ├── refactor_plan-update.md      # Full refactor plan (this file)
│   ├── dev_environment_setup.md     # Development environment guide
│   └── refactor_changelog.md        # Log of duplicated functions & tests
│
├── examples/                         # Existing examples (refactored gradually)
│   ├── Basic calculations PCA and PLS/
│   ├── ...
│   └── (update imports as modules are completed)
│
├── pyproject.toml                    # Poetry configuration
├── poetry.lock                       # Locked dependencies
├── pytest.ini                        # Pytest configuration
├── setup.py                          # Setup configuration
├── requirements.txt                  # Keep for compatibility
├── README.md
└── LICENSE
```

---

## 🔄 Refactoring Workflow (TDD Approach)

For **every function** listed below:

1. **Write Tests First (RED)**: Create comprehensive tests in `tests/test_*.py`
   ```bash
   pytest tests/test_utils.py::test_meancenterscale -v
   # Should fail (function doesn't exist yet in module)
   ```

2. **Copy the Function (GREEN)**: Duplicate the implementation from `pyphi.py` into the new module. Do **not** delete or modify the original yet.

3. **Update Imports**: Ensure the function is exposed from the new module (and eventually `src/pyphi/__init__.py` when appropriate) while leaving all existing `pyphi.py` usage untouched for now.

4. **Run Tests**: 
   ```bash
   pytest tests/test_utils.py::test_meancenterscale -v
   # Should pass
   ```

5. **Integration Test**: Run an example script to verify backward compatibility:
   ```bash
   python examples/Basic\ calculations\ PCA\ and\ PLS/Example_Script.py
   ```

6. **Record the change**: Document the duplication in `notes/refactor_changelog.md`. Removal from `pyphi.py` will happen in a later cleanup phase once all modules have been migrated.

7. **Commit**: 
   ```bash
   git commit -m "refactor(utils): duplicate meancenterscale into utils.py

   - Tests pass: test_utils.py::test_meancenterscale
   - Example script verified
   - Legacy implementation left in pyphi.py for compatibility"
   ```

---

## 🎯 Module Organization & Function Allocation

### Phase 0: Project Setup (BEFORE any code moves)

**Create empty test files and structure:**

```bash
mkdir -p src/pyphi
mkdir -p tests
touch tests/__init__.py
touch tests/conftest.py
touch tests/test_{utils,spectra,pca,pls,advanced_pls,diagnostics,exports}.py
touch src/pyphi/{__init__,_internal,utils,spectra,pca,pls,advanced_pls,diagnostics,exports}.py
```

**Create pytest.ini:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

**Update pyproject.toml:**
- Update `packages` configuration to point to `src/pyphi`
- Add `pytest` as dev dependency
- Add test entry point

---

### Phase 1: `src/pyphi/utils.py` - Core Utilities (Duplication Pass)

**Dependencies**: None (foundation layer)

In this phase we **duplicate** the utility helpers from the monolithic `pyphi.py` into the modular package. The originals stay in place so the legacy API remains untouched until the consolidation phase.

- Create `src/pyphi/utils.py` and copy the implementations verbatim from `pyphi.py`.
- Update `src/pyphi/__init__.py` to re-export utilities once their downstream dependencies exist (optional during early steps).
- Log each duplicated function in `notes/refactor_changelog.md` with a short description and test reference.

#### Public Functions to duplicate now:
- [ ] `clean_htmls()` - Remove HTML files from directory
- [ ] `clean_empty_rows()` - Remove empty rows from data matrix
- [ ] `clean_low_variances()` - Remove low-variance columns
- [ ] `find()` - Find indices matching a condition
- [ ] `unique()` - Get unique values from DataFrame column
- [ ] `isin_ordered_col0()` - Check if values are in ordered column
- [ ] `reconcile_rows()` - Align rows across multiple DataFrames
- [ ] `reconcile_rows_to_columns()` - Align rows to column identifiers

#### Private / Semi-Private Functions (still duplicated now; moved to `_internal.py` later):
- [ ] `mean()` - Calculate mean (handles NaN)
- [ ] `std()` - Calculate standard deviation (handles NaN)
- [ ] `z2n()` - Convert zeros back to NaN
- [ ] `n2z()` - Convert NaN to zeros for computation
- [ ] `scores_conf_int_calc()` - Helper for confidence interval calculation
- [ ] `ma57_dummy_check()` - Check IPOPT MA57 availability (keep in `__init__.py` later)
- [ ] `meancenterscale()` - Mean center and/or scale data
- [ ] `np2D2pyomo()` - Convert 2D NumPy array to Pyomo format
- [ ] `np1D2pyomo()` - Convert 1D NumPy array to Pyomo format
- [ ] `spe_ci()` - SPE confidence interval calculation
- [ ] `single_score_conf_int()` - Score confidence interval calculation
- [ ] `f99()` - F-distribution 99th percentile
- [ ] `f95()` - F-distribution 95th percentile
- [ ] `_Ab_btbinv()` - Matrix projection helper (move to `_internal.py` during consolidation)

> **Note:** Once all modules are duplicated and verified, a later "Consolidation" milestone will replace the legacy implementations in `pyphi.py` with thin wrappers or full removals.

**Test Strategy:**
- Unit tests for each duplicated function with edge cases (NaN, missing data, etc.)
- Property-based tests for mathematical functions
- Integration tests with the core models later

---

### Phase 2: `src/pyphi/spectra.py` - Spectral Preprocessing

**Dependencies**: `utils.py` (may use for scaling)

#### Public Functions:
- [ ] `spectra_snv()` - Standard Normal Variate preprocessing
- [ ] `spectra_savgol()` - Savitzky-Golay filter
- [ ] `spectra_mean_center()` - Mean center spectra
- [ ] `spectra_autoscale()` - Autoscale spectra
- [ ] `spectra_baseline_correction()` - Baseline correction
- [ ] `spectra_msc()` - Multiplicative Scatter Correction

**Test Strategy:**
- Test with real spectral data (NIR example)
- Verify preprocessing doesn't corrupt data
- Integration test with spectral PLS example

---

### Phase 3: `src/pyphi/pca.py` - Principal Component Analysis

**Dependencies**: `utils.py`, `_internal.py`

#### Public Functions:
- [ ] `pca()` - Main PCA function (wrapper with cross-validation)
- [ ] `pca_()` - Core PCA algorithm (RENAME from `pca_` to `_pca_core()` or keep internal)

#### Private Functions (in `_internal.py`):
- [ ] `pca_()` - Can be marked as internal or kept in pca.py as helper

**Notes:**
- `pca()` calls `pca_()` - keep them together
- Handles missing data via NIPALS or NLP
- Complex Pyomo integration for NLP algorithm

**Test Strategy:**
- Test PCA with complete data (SVD vs NIPALS)
- Test PCA with missing data (NLP algorithm)
- Cross-validation tests
- Compare results with sklearn PCA
- Integration: Example_Script.py uses `phi.pca()`

---

### Phase 4: `src/pyphi/pls.py` - Partial Least Squares

**Dependencies**: `utils.py`, `_internal.py`, `pca.py` (optional)

#### Public Functions:
- [ ] `pls()` - Main PLS function (wrapper with cross-validation)
- [ ] `pls_()` - Core PLS algorithm
- [ ] `pls_pred()` - PLS prediction on new data
- [ ] `pls_cca()` - CCA calculation for PLS (helper for OPLS-like behavior)

#### Semi-Private:
- [ ] `pls_cca()` - Only used internally by `pls()` when `cca=True`, but might keep public

**Notes:**
- Very similar structure to PCA but for Y prediction
- Handles both univariate and multivariate Y
- CCA integration for orthogonal modeling

**Test Strategy:**
- Test basic PLS with univariate Y
- Test multivariate Y prediction
- Cross-validation tests
- CCA flag tests
- Integration: Example_Script.py uses `phi.pls()`

---

### Phase 5: `src/pyphi/advanced_pls.py` - Advanced Models

**Dependencies**: `utils.py`, `_internal.py`, `pls.py`, `pca.py`

These are specialized model structures built on PLS/PCA foundations.

#### Public Functions:
- [ ] `lwpls()` - Locally Weighted PLS
- [ ] `mbpls()` - Multi-block PLS
- [ ] `lpls()` - Linear PLS (bilinear model)
- [ ] `lpls_pred()` - LPLS prediction
- [ ] `jrpls()` - Joint Range PLS
- [ ] `jrpls_pred()` - JRPLS prediction
- [ ] `tpls()` - Tensor/Three-way PLS
- [ ] `tpls_pred()` - TPLS prediction

**Notes:**
- Each function has associated `_pred()` function for predictions
- JRPLS and TPLS handle multiple X matrices (arrays of matrices)
- All use `meancenterscale()` and `_Ab_btbinv()`
- Complex algorithm implementations

**Test Strategy:**
- Test each model type with synthetic data
- Verify predictions on held-out data
- Integration tests with corresponding examples (if they exist)

---

### Phase 6: `src/pyphi/diagnostics.py` - Diagnostics & Analysis Tools

**Dependencies**: All other modules (analysis on model objects)

#### Public Functions:
- [ ] `hott2()` - Hotelling T² statistic
- [ ] `spe()` - Squared Prediction Error
- [ ] `contributions()` - Contribution analysis
- [ ] `varimax_()` - Varimax rotation algorithm (INTERNAL)
- [ ] `varimax_rotation()` - Apply varimax to model
- [ ] `bootstrap_pls()` - Bootstrap PLS for uncertainty
- [ ] `bootstrap_pls_pred()` - Bootstrap predictions
- [ ] `build_polynomial()` - Polynomial regression with PLS-aided variable selection
- [ ] `cca()` - Canonical Correlation Analysis
- [ ] `cca_multi()` - Multi-response CCA

#### Helper Functions (move to `_internal.py`):
- [ ] `findstr()` - Find operator positions in string
- [ ] `evalvar()` - Evaluate variable expressions
- [ ] `writeeq()` - Format equation string
- [ ] `varimax_()` - Internal rotation algorithm

**Notes:**
- These operate on model dictionary objects from PCA/PLS
- No circular dependencies (models are simple dicts)
- Contributions are computationally intensive

**Test Strategy:**
- Test each diagnostic with known model objects
- Test polynomial building with sample expressions
- Verify contributions match manual calculations
- Integration: Varimax Rotation example

---

### Phase 7: `src/pyphi/exports.py` - I/O & Model Export

**Dependencies**: All other modules

#### Public Functions:
- [ ] `export_2_gproms()` - Export model to gPROMS format
- [ ] `adapt_pls_4_pyomo()` - Adapt PLS model for Pyomo optimization
- [ ] `prep_pca_4_MDbyNLP()` - Prepare PCA for MD by NLP
- [ ] `prep_pls_4_MDbyNLP()` - Prepare PLS for MD by NLP
- [ ] `parse_materials()` - Parse materials from spreadsheet
- [ ] `cat_2_matrix()` - Convert categorical data to matrix
- [ ] `conv_pls_2_eiot()` - Convert PLS to EIOT format (internal/specialized)

#### Semi-Private:
- [ ] `conv_pls_2_eiot()` - Specialized conversion (maybe internal?)

**Test Strategy:**
- Test export formats with sample models
- Verify round-trip consistency where applicable
- Integration: Verify exports work with actual gPROMS/Pyomo

---

### Phase 8: `src/pyphi/_internal.py` - Private Helpers

**ALL INTERNAL FUNCTIONS GO HERE** - Functions starting with `_` or only used internally

```python
# Functions from various modules
_mean()
_std()
_z2n()
_n2z()
_Ab_btbinv()
_varimax_()
_scores_conf_int_calc()
_ma57_dummy_check()
_meancenterscale()
_np2D2pyomo()
_np1D2pyomo()
_spe_ci()
_single_score_conf_int()
_f99()
_f95()
_findstr()
_evalvar()
_writeeq()
# ... any other internal helpers
```

**Benefits:**
- Clear separation of private vs public API
- Easier to refactor internals without breaking user code
- IDE autocomplete won't clutter with internal functions
- Clear ownership of helper functions

---

### Phase 9: `src/pyphi/__init__.py` - Public API & Configuration

**This is CRITICAL for backward compatibility and user experience.**

```python
"""
PyPhi - Multivariate Analysis Toolbox
"""

import os
import logging

# ============================================================================
# PYOMO & SOLVER CONFIGURATION (must happen early)
# ============================================================================

os.environ['NEOS_EMAIL'] = 'pyphisoftware@gmail.com'

# Conditional imports for solver detection
try:
    from pyomo.environ import *
    pyomo_ok = True
except ImportError:
    pyomo_ok = False

from shutil import which

if bool(which('gams')):
    gams_ok = True
else:
    gams_ok = False

ipopt_ok = bool(which('ipopt'))

# Check GAMS interface
if pyomo_ok and gams_ok:
    try:
        from pyomo.solvers.plugins.solvers.GAMS import GAMSDirect, GAMSShell
        gams_ok = (GAMSDirect().available(exception_flag=False)
                   or GAMSShell().available(exception_flag=False))
    except:
        gams_ok = False

# Check MA57 availability
ma57_ok = False
if pyomo_ok and ipopt_ok:
    from ._internal import _ma57_dummy_check
    ma57_ok = _ma57_dummy_check()

if not(pyomo_ok) or (not(ipopt_ok) and not(gams_ok)):
    logging.warning('Will be using the NEOS server in the absence of IPOPT and GAMS')

# ============================================================================
# PUBLIC API - Import everything users need
# ============================================================================

# Core models
from .pca import pca, pca_
from .pls import pls, pls_, pls_pred, pls_cca
from .advanced_pls import (
    lwpls, mbpls, lpls, lpls_pred, jrpls, jrpls_pred, tpls, tpls_pred
)

# Data preprocessing & utilities
from .utils import (
    clean_htmls, clean_empty_rows, clean_low_variances,
    find, unique, isin_ordered_col0, reconcile_rows, reconcile_rows_to_columns
)

# Spectral preprocessing
from .spectra import (
    spectra_snv, spectra_savgol, spectra_mean_center,
    spectra_autoscale, spectra_baseline_correction, spectra_msc
)

# Diagnostics & analysis
from .diagnostics import (
    hott2, spe, contributions, varimax_rotation,
    bootstrap_pls, bootstrap_pls_pred,
    build_polynomial, cca, cca_multi
)

# Import/Export
from .exports import (
    export_2_gproms, adapt_pls_4_pyomo,
    prep_pca_4_MDbyNLP, prep_pls_4_MDbyNLP,
    parse_materials, cat_2_matrix
)

# ============================================================================
# DEFINE PUBLIC API
# ============================================================================

__version__ = "5.0.0"
__author__ = "Salvador Garcia-Munoz (sgarciam@ic.ac.uk)"

__all__ = [
    # PCA
    'pca', 'pca_',
    # PLS
    'pls', 'pls_', 'pls_pred', 'pls_cca',
    # Advanced PLS
    'lwpls', 'mbpls', 'lpls', 'lpls_pred',
    'jrpls', 'jrpls_pred', 'tpls', 'tpls_pred',
    # Utilities
    'clean_htmls', 'clean_empty_rows', 'clean_low_variances',
    'find', 'unique', 'isin_ordered_col0', 'reconcile_rows',
    'reconcile_rows_to_columns',
    # Spectra
    'spectra_snv', 'spectra_savgol', 'spectra_mean_center',
    'spectra_autoscale', 'spectra_baseline_correction', 'spectra_msc',
    # Diagnostics
    'hott2', 'spe', 'contributions', 'varimax_rotation',
    'bootstrap_pls', 'bootstrap_pls_pred',
    'build_polynomial', 'cca', 'cca_multi',
    # Exports
    'export_2_gproms', 'adapt_pls_4_pyomo',
    'prep_pca_4_MDbyNLP', 'prep_pls_4_MDbyNLP',
    'parse_materials', 'cat_2_matrix',
    # Metadata
    '__version__', '__author__',
]

# Solver status
__solver_status__ = {
    'pyomo_ok': pyomo_ok,
    'gams_ok': gams_ok,
    'ipopt_ok': ipopt_ok,
    'ma57_ok': ma57_ok,
}
```

---

## 📝 Implementation Strategy

### Step-by-step execution:

1. **Setup Phase** (Before Phase 1)
   - [ ] Create `src/pyphi/` directory structure
   - [ ] Create `tests/` directory with test files
   - [ ] Create `pytest.ini`
   - [ ] Update `pyproject.toml` with new structure
   - [ ] Create `conftest.py` with shared fixtures
   - [ ] Create `notes/refactor_changelog.md` to log duplicated functions and test coverage
   - Commit: `chore: initialize package structure and test framework`

2. **Phase 1-9**: Execute sequentially
   - For each phase, follow the TDD workflow above
   - One commit per function (or logical group)
   - Run tests locally before committing
   - Run integration tests periodically

3. **Consolidation / Backward Compatibility Phase**
   - [ ] After every function lives under `src/pyphi`, migrate consumers to import from the package instead of the legacy `pyphi.py`
   - [ ] Once no code depends on the monolithic module, archive or delete `pyphi.py` and rely exclusively on the new package layout
   - [ ] Verify all examples work using the package imports
   - Commit: `chore: finalize modular package`

4. **Final Cleanup**
   - [ ] Run full test suite
   - [ ] Update documentation to reference new package structure
   - [ ] Create migration guide for contributors
   - [ ] Consider deprecation warnings if function names changed
   - Commit: `docs: update docs for new package structure`

---

## 🧪 Testing Strategy

### Test Files Structure

```python
# tests/conftest.py - Shared fixtures
import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    """Generate consistent sample data for all tests"""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    Y = np.random.randn(50, 3)
    return X, Y

@pytest.fixture
def sample_dataframe(sample_data):
    """Convert to pandas DataFrame with observation IDs"""
    X, Y = sample_data
    X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
    X_df.insert(0, 'ObsID', [f'Obs{i}' for i in range(X.shape[0])])
    return X_df

@pytest.fixture
def missing_data():
    """Generate data with missing values"""
    X = np.random.randn(50, 10)
    # Introduce some NaNs
    X[np.random.rand(*X.shape) < 0.1] = np.nan
    return X
```

### Test Execution:

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_utils.py -v

# Run with coverage
pytest tests/ --cov=src/pyphi --cov-report=html

# Run integration tests (examples)
python examples/Basic\ calculations\ PCA\ and\ PLS/Example_Script.py
```

---

## 🔗 Function Dependencies Map

```
_internal.py (no dependencies)
    ↓
utils.py (uses _internal.py)
    ↓
spectra.py (uses utils.py)
    ↓
pca.py (uses utils.py, _internal.py)
    ↓
pls.py (uses utils.py, _internal.py, pca.py optional)
    ↓
advanced_pls.py (uses pca.py, pls.py, utils.py, _internal.py)
    ↓
diagnostics.py (uses all above - but only takes model dicts as input, no circular deps)
    ↓
exports.py (uses all above)
```

**NO CIRCULAR DEPENDENCIES** - This structure is safe!

---

## 📦 Git Commit Strategy

Use conventional commits for clear history:

```
# Initial setup
chore: initialize package structure

# For each function/module move
refactor(MODULE): move FUNCTION_NAME to MODULE.py

  - Implement FUNCTION_NAME in MODULE.py
  - Add comprehensive tests in test_MODULE.py
  - Verified with integration test: EXAMPLE_NAME
  - Removed from pyphi.py
  
  Closes: #XX (issue number if applicable)

# Example:
refactor(utils): move meancenterscale to utils.py

  - Implement meancenterscale() in utils.py
  - Add tests for normal/center/autoscale modes
  - Test missing data handling
  - Verified with Example_Script.py
  - Removed from pyphi.py

test(utils): add comprehensive tests for mean/std

  - Add tests for NaN handling
  - Add tests for column-wise calculations
  - Property tests for mathematical correctness

# Final cleanup
chore: remove deprecated pyphi.py monolith
docs: update README with new package structure
```

---

## ✅ Verification Checklist

After each phase:

- [ ] All tests pass: `pytest tests/test_MODULE.py -v`
- [ ] Full test suite passes: `pytest tests/ -v`
- [ ] At least one example runs without modification
- [ ] No import errors when running examples
- [ ] Functions are properly exported in `__init__.py`
- [ ] Help text works: `help(pyphi.FUNCTION_NAME)`
- [ ] Type hints are accurate (if added)

Before finalizing:

- [ ] All examples run successfully
- [ ] Full test coverage > 80%
- [ ] Documentation is updated
- [ ] Performance is comparable to original
- [ ] No new warnings or errors on import
- [ ] Backward compatibility maintained

---

## 📚 Examples Update Schedule

Update examples as modules are completed:

**After Phase 1 (utils):**
- All examples should work (utils are imported internally)

**After Phase 3 (pca):**
- `examples/Basic\ calculations\ PCA\ and\ PLS/Example_Script.py` ✓

**After Phase 4 (pls):**
- All examples should work ✓

**After Phase 6 (diagnostics):**
- `examples/Varimax\ Rotation/chem_exp_example.py` ✓

**After Phase 7 (exports):**
- Verify any export-dependent examples

---

## 🎯 Success Criteria

The refactor is complete when:

1. ✅ All functions moved to appropriate modules
2. ✅ All tests pass (>80% coverage)
3. ✅ All examples run without modification
4. ✅ `import pyphi as phi; phi.pca(...)` works exactly as before
5. ✅ New structure is documented
6. ✅ Git history is clean with meaningful commits
7. ✅ Performance is not degraded
8. ✅ Private API is clearly marked (leading `_`)
9. ✅ Public API is documented in `__init__.py`
10. ✅ Future contributors understand the structure

---

## 📖 Additional Notes

### Why Option B?

- Cleaner structure from day one
- Supports future growth and plugins
- Proper separation of concerns
- Easier testing and CI/CD
- Industry-standard layout
- Supports PyPI publication

### Backward Compatibility

The key to maintaining backward compatibility:
- `__init__.py` imports and re-exports everything
- Users never import from internal modules
- All public function signatures preserved
- All public function behavior preserved

### Private vs Public

**Public** (in `__all__`):
- `pca`, `pls`, `pca_`, `pls_`
- All spectral functions
- Diagnostics functions
- Export functions

**Internal** (leading `_`, not in `__all__`):
- `_mean`, `_std`
- `_Ab_btbinv`
- `_varimax_`
- `_findstr`, `_evalvar`, `_writeeq`
- `_meancenterscale`
- `_np2D2pyomo`, `_np1D2pyomo`
- Helper functions that users shouldn't call directly

### When to Make Functions Public vs Private

**Keep Public if:**
- User-facing function documented in examples
- Part of official API
- Users can reasonably call it independently
- Examples: `pca()`, `pls()`, `spectra_snv()`, `build_polynomial()`

**Make Private if:**
- Internal helper for other functions
- Implementation detail that could change
- Users shouldn't need to call directly
- Examples: `_meancenterscale()`, `_Ab_btbinv()`, `_mean()`, `_std()`

---

## 🚀 Next Steps

1. ✅ Review this refined plan
2. Start with **Setup Phase** to create directory structure
3. Begin **Phase 1** with `_internal.py` and `utils.py`
4. Follow the TDD workflow strictly
5. Commit frequently with clear messages
6. Test integrations early and often
7. Document as you go

