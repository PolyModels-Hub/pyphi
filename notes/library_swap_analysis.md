# PyPhi Library Swap Analysis

## Overview

This document analyzes custom functions in `pyphi_legacy.py` that can be replaced with well-maintained external library equivalents. Swapping these functions reduces maintenance burden, improves reliability, and leverages optimized implementations.

**Current Dependencies:** NumPy, SciPy, Pandas, Matplotlib, Pyomo, Bokeh, openpyxl, xlrd

---

## High-Priority Replacements

These functions have direct, well-tested library equivalents that should be used instead of custom implementations.

### 1. `mean()` → `np.nanmean()`

**Current Implementation** (`pyphi_legacy.py:2197-2208`):
```python
def mean(X):
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map = X_nan_map * 1
        X_[X_nan_map == 1] = 0
        aux = np.sum(X_nan_map, axis=0)
        x_mean = np.sum(X_, axis=0, keepdims=1) / (np.ones((1, X_.shape[1])) * X_.shape[0] - aux)
    else:
        x_mean = np.mean(X_, axis=0, keepdims=1)
    return x_mean
```

**Replacement:**
```python
def mean(X):
    return np.nanmean(X, axis=0, keepdims=True)
```

**Benefits:**
- NumPy's `nanmean` is heavily optimized (C-level implementation)
- Handles NaN values correctly by design
- Less code to maintain
- Better numerical stability

**Migration Notes:**
- Behavior is identical for the expected use cases
- Test edge cases: all-NaN columns, empty arrays

---

### 2. `std()` → `np.nanstd()`

**Current Implementation** (`pyphi_legacy.py:2210-2225`):
```python
def std(X):
    x_mean = mean(X)
    x_mean = np.tile(x_mean, (X.shape[0], 1))
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map = X_nan_map * 1
        X_ = X.copy()
        X_[X_nan_map == 1] = 0
        aux_mat = (X_ - x_mean) ** 2
        aux_mat[X_nan_map == 1] = 0
        aux = np.sum(X_nan_map, axis=0)
        x_std = np.sqrt((np.sum(aux_mat, axis=0, keepdims=1)) / (np.ones((1, X_.shape[1])) * (X_.shape[0] - 1 - aux)))
    else:
        x_std = np.sqrt(np.sum((X - x_mean) ** 2, axis=0, keepdims=1) / (np.ones((1, X.shape[1])) * (X.shape[0] - 1)))
    return x_std
```

**Replacement:**
```python
def std(X):
    return np.nanstd(X, axis=0, keepdims=True, ddof=1)
```

**Benefits:**
- Single function call vs 15+ lines of code
- `ddof=1` parameter gives sample standard deviation (Bessel's correction)
- Optimized C implementation
- Handles edge cases automatically

**Migration Notes:**
- The custom implementation uses `ddof=1` (sample std), ensure `ddof=1` is set in replacement
- Test with various NaN patterns

---

### 3. `f95()` and `f99()` → `scipy.stats.f.ppf()`

**Current Implementation** (`pyphi_legacy.py:2565-2656`):
Uses hardcoded lookup tables with 2D interpolation for F-distribution critical values.

**Replacement:**
```python
from scipy.stats import f as f_dist

def f95(dfn, dfd):
    """F-distribution 95th percentile (critical value for alpha=0.05)"""
    return f_dist.ppf(0.95, dfn, dfd)

def f99(dfn, dfd):
    """F-distribution 99th percentile (critical value for alpha=0.01)"""
    return f_dist.ppf(0.99, dfn, dfd)
```

**Benefits:**
- Exact calculation vs interpolated lookup table
- Works for any degrees of freedom (not limited to table values)
- No hardcoded arrays to maintain
- Numerically accurate for edge cases

**Migration Notes:**
- Results may differ slightly due to interpolation vs exact calculation
- Update tests to use appropriate tolerances
- Consider adding a thin wrapper for backward compatibility if needed

---

### 4. `spe_ci()` → `scipy.stats.chi2.ppf()`

**Current Implementation** (`pyphi_legacy.py:2463-2518`):
Uses hardcoded chi-squared table with interpolation.

**Replacement:**
```python
from scipy.stats import chi2

def spe_ci(spe):
    """Calculate SPE confidence intervals using chi-squared distribution."""
    spe_mean = np.mean(spe)
    if spe_mean > 1e-16:
        spe_var = np.var(spe, ddof=1)
        g = spe_var / (2 * spe_mean)
        h = (2 * spe_mean ** 2) / spe_var
        lim95 = g * chi2.ppf(0.95, h)
        lim99 = g * chi2.ppf(0.99, h)
    else:
        lim95 = 0
        lim99 = 0
    return lim95, lim99
```

**Benefits:**
- Exact chi-squared quantiles vs table interpolation
- Works for any degrees of freedom
- Removes ~40 lines of hardcoded table data
- More accurate for non-integer degrees of freedom

---

### 5. `single_score_conf_int()` → `scipy.stats.t.ppf()`

**Current Implementation** (`pyphi_legacy.py:2520-2563`):
Uses hardcoded t-distribution table with interpolation.

**Replacement:**
```python
from scipy.stats import t as t_dist

def single_score_conf_int(scores):
    """Calculate confidence intervals for scores using t-distribution."""
    n = scores.shape[0]
    st = np.var(scores, ddof=1)
    lim95 = t_dist.ppf(0.975, n - 1) * np.sqrt(st)  # Two-tailed 95%
    lim99 = t_dist.ppf(0.995, n - 1) * np.sqrt(st)  # Two-tailed 99%
    return lim95, lim99
```

**Benefits:**
- Exact t-distribution quantiles
- Works for any sample size
- Removes ~35 lines of hardcoded table data

**Migration Notes:**
- Note the use of 0.975 for two-tailed 95% CI
- Verify the original implementation's intended behavior (one-tailed vs two-tailed)

---

### 6. `spectra_savgol()` → `scipy.signal.savgol_filter()`

**Current Implementation** (`pyphi_legacy.py:2298-2356`):
Custom Savitzky-Golay filter implementation building the transformation matrix manually.

**Replacement:**
```python
from scipy.signal import savgol_filter, savgol_coeffs

def spectra_savgol(ws, od, op, Dm):
    """
    Apply Savitzky-Golay filter to spectra.
    
    Args:
        ws: Half window size (full window = 2*ws + 1)
        od: Order of derivative
        op: Order of polynomial
        Dm: Spectra data (DataFrame or ndarray)
    
    Returns:
        Dm_sg: Filtered spectra
        M: Transformation matrix (for applying to new samples)
    """
    window_length = 2 * ws + 1
    
    if isinstance(Dm, pd.DataFrame):
        # Handle DataFrame case
        x_columns = Dm.columns.tolist()
        first_col = [x_columns[0]]
        first_col.extend(x_columns[1:][ws:-ws])
        x_values = Dm.values
        col1 = Dm.values[:, 0].reshape(-1, 1)
        filtered, M = spectra_savgol(ws, od, op, x_values[:, 1:].astype(float))
        data_ = np.hstack((col1, filtered))
        return pd.DataFrame(data=data_, columns=first_col), M
    else:
        # Apply scipy's savgol_filter
        if Dm.ndim == 1:
            Dm_sg = savgol_filter(Dm, window_length, op, deriv=od, mode='interp')
            # Trim to match original behavior
            Dm_sg = Dm_sg[ws:-ws]
        else:
            Dm_sg = savgol_filter(Dm, window_length, op, deriv=od, axis=1, mode='interp')
            Dm_sg = Dm_sg[:, ws:-ws]
        
        # Build transformation matrix for compatibility
        coeffs = savgol_coeffs(window_length, op, deriv=od)
        l = Dm.shape[1] if Dm.ndim == 2 else Dm.shape[0]
        M = _build_savgol_matrix(coeffs, l, ws)
        
        return Dm_sg, M
```

**Benefits:**
- SciPy's implementation is optimized and well-tested
- Handles edge modes (constant, nearest, wrap, interp)
- Better numerical stability
- Removes ~60 lines of custom matrix construction code

**Migration Notes:**
- The window size convention differs: scipy uses full window, original uses half-window
- May need wrapper to maintain API compatibility
- The transformation matrix `M` is still needed for applying to new samples

---

### 7. `find()` → `np.where()` or `np.argwhere()`

**Current Implementation** (`pyphi_legacy.py:2932-2933`):
```python
def find(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]
```

**Replacement:**
```python
# Option 1: Direct replacement with np.where
def find(a, func):
    return np.where(np.vectorize(func)(a))[0].tolist()

# Option 2: For simple conditions, use np.where directly
# Instead of: find(x, lambda v: v > 5)
# Use: np.where(x > 5)[0]
```

**Benefits:**
- NumPy's vectorized operations are much faster for large arrays
- Standard idiom recognizable to NumPy users

**Migration Notes:**
- The custom `find()` accepts any callable; replacement may need adjustment
- Consider deprecating and updating call sites to use `np.where()` directly

---

### 8. `unique()` → `pd.unique()`

**Current Implementation** (`pyphi_legacy.py:3620-3640`):
```python
def unique(df, colid):
    """Returns unique values in the column of a DataFrame in order of occurrence"""
    # Custom implementation preserving order
```

**Replacement:**
```python
def unique(df, colid):
    """Returns unique values in the column of a DataFrame in order of occurrence."""
    return pd.unique(df[colid]).tolist()
```

**Benefits:**
- Pandas' `unique()` preserves order of first occurrence (unlike `np.unique`)
- Single function call
- Well-documented behavior

---

## Medium-Priority Replacements

These functions could be replaced but require more consideration due to API differences or added dependencies.

### 9. `meancenterscale()` → `sklearn.preprocessing.StandardScaler`

**Current Implementation** (`pyphi_legacy.py:2227-2261`):
Custom function supporting three modes: both (True), center-only, autoscale-only.

**Potential Replacement:**
```python
from sklearn.preprocessing import StandardScaler

def meancenterscale(X, *, mcs=True):
    """Mean center and/or scale a matrix."""
    if isinstance(mcs, bool):
        if mcs:
            scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            return X, np.nan, np.nan
    elif mcs == 'center':
        scaler = StandardScaler(with_mean=True, with_std=False)
    elif mcs == 'autoscale':
        scaler = StandardScaler(with_mean=False, with_std=True)
    else:
        return X, np.nan, np.nan
    
    X_scaled = scaler.fit_transform(X)
    x_mean = scaler.mean_.reshape(1, -1) if scaler.with_mean else np.zeros((1, X.shape[1]))
    x_std = scaler.scale_.reshape(1, -1) if scaler.with_std else np.ones((1, X.shape[1]))
    
    return X_scaled, x_mean, x_std
```

**Considerations:**
- **Pros:** Well-tested, handles edge cases, supports incremental learning
- **Cons:** 
  - Adds sklearn as a dependency (currently not required)
  - Different NaN handling (sklearn doesn't handle NaN by default)
  - API differences may cause issues

**Recommendation:** Keep custom implementation OR use numpy's `nanmean`/`nanstd` directly:
```python
def meancenterscale(X, *, mcs=True):
    if isinstance(mcs, bool) and mcs:
        x_mean = np.nanmean(X, axis=0, keepdims=True)
        x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
        X_scaled = (X - x_mean) / x_std
    elif mcs == 'center':
        x_mean = np.nanmean(X, axis=0, keepdims=True)
        x_std = np.ones((1, X.shape[1]))
        X_scaled = X - x_mean
    elif mcs == 'autoscale':
        x_mean = np.zeros((1, X.shape[1]))
        x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
        X_scaled = X / x_std
    else:
        return X, np.nan, np.nan
    return X_scaled, x_mean, x_std
```

---

### 10. `spectra_snv()` - Keep Custom (with simplification)

**Current Implementation** (`pyphi_legacy.py:2263-2296`):
Row-wise Standard Normal Variate transformation.

**Analysis:**
SNV is a domain-specific operation (row-wise standardization for spectral data). While it can be expressed using sklearn or numpy, the current implementation is clear and serves its purpose.

**Simplified Implementation:**
```python
def spectra_snv(x):
    """Row-wise Standard Normal Variate transform for spectroscopic data."""
    if isinstance(x, pd.DataFrame):
        x_values = x.values[:, 1:].astype(float)
        x_transformed = spectra_snv(x_values)
        result = x.copy()
        result.iloc[:, 1:] = x_transformed
        return result
    else:
        if x.ndim == 2:
            mean_x = np.mean(x, axis=1, keepdims=True)
            std_x = np.std(x, axis=1, keepdims=True, ddof=1)
            return (x - mean_x) / std_x
        else:
            return (x - np.mean(x)) / np.std(x, ddof=1)
```

**Recommendation:** Keep custom but simplify using standard numpy functions.

---

### 11. `varimax_()` → External Rotation Library

**Current Implementation** (`pyphi_legacy.py:5298-5309`):
```python
def varimax_(X, gamma=1.0, q=20, tol=1e-6):
    p, k = X.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_ = d
        Lambda = dot(X, R)
        u, s, vh = svd(dot(X.T, asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d_ != 0 and d/d_ < 1 + tol:
            break
    return dot(X, R)
```

**Potential Replacement:**
```python
# Option 1: factor_analyzer package (adds dependency)
from factor_analyzer.rotator import Rotator
rotator = Rotator(method='varimax')
rotated = rotator.fit_transform(loadings)

# Option 2: scipy's ortho_group for orthogonal rotations (partial)
# Not a direct equivalent - varimax is a specific rotation criterion
```

**Considerations:**
- `factor_analyzer` package provides varimax but adds a dependency
- The current implementation is compact and well-understood
- Custom implementation allows `gamma` parameter for promax-like behavior

**Recommendation:** Keep custom implementation. It's only ~12 lines and well-tested in the chemometrics domain.

---

## Functions to Keep Custom

These functions should remain custom implementations.

### `n2z()` and `z2n()` - Simple NaN Utilities
- No standard equivalent exists
- Only 6 lines total
- Clear, simple purpose

### `hott2()` - Hotelling's T² Statistic
- Domain-specific diagnostic
- No direct sklearn/scipy equivalent
- Integrates with PyPhi's model structure

### `spe()` - Squared Prediction Error
- Domain-specific diagnostic
- Depends on PyPhi's `pls_pred` function
- Keep as-is

### `contributions()` - Contribution Analysis
- Domain-specific chemometric function
- Complex logic for different contribution types
- No standard equivalent

### `np2D2pyomo()` / `np1D2pyomo()` - Pyomo Converters
- Pyomo-specific utilities
- Required for NLP optimization functionality
- No standard equivalent

### Core PCA/PLS Algorithms
- `pca_()` and `pls_()` implement NIPALS with missing data handling
- The NLP-based missing data approach is unique to PyPhi
- sklearn's PCA/PLS don't handle missing data the same way
- Keep these as the core value proposition of PyPhi

---

## Changes to Refactor Plan

### Update Phase 1: `src/pyphi/utils.py`

**Modified function list:**

| Original Function | Action | Implementation |
|------------------|--------|----------------|
| `mean()` | Replace | Use `np.nanmean()` |
| `std()` | Replace | Use `np.nanstd(ddof=1)` |
| `meancenterscale()` | Simplify | Use numpy nan-functions internally |
| `find()` | Replace | Use `np.where()` wrapper or deprecate |
| `unique()` | Replace | Use `pd.unique()` |
| `n2z()` | Keep | Simple utility |
| `z2n()` | Keep | Simple utility |
| `f95()` | Replace | Use `scipy.stats.f.ppf(0.95, ...)` |
| `f99()` | Replace | Use `scipy.stats.f.ppf(0.99, ...)` |
| `spe_ci()` | Replace | Use `scipy.stats.chi2.ppf()` |
| `single_score_conf_int()` | Replace | Use `scipy.stats.t.ppf()` |
| `scores_conf_int_calc()` | Simplify | Update to use scipy.stats |

### Update Phase 2: `src/pyphi/spectra.py`

| Original Function | Action | Implementation |
|------------------|--------|----------------|
| `spectra_savgol()` | Replace | Use `scipy.signal.savgol_filter()` |
| `spectra_snv()` | Simplify | Use `np.mean()`/`np.std()` directly |
| `spectra_mean_center()` | Simplify | Use `np.nanmean()` |
| `spectra_autoscale()` | Simplify | Use `np.nanstd()` |
| `spectra_baseline_correction()` | Keep | Simple operation |
| `spectra_msc()` | Keep | Domain-specific |

### Update Phase 6: `src/pyphi/diagnostics.py`

| Original Function | Action | Implementation |
|------------------|--------|----------------|
| `varimax_()` | Keep | Custom implementation adequate |
| `hott2()` | Keep | Domain-specific |
| `spe()` | Keep | Domain-specific |
| `contributions()` | Keep | Domain-specific |

---

## Dependency Considerations

### No New Dependencies Required

All recommended replacements use libraries already in `requirements.txt`:
- **NumPy** (already required): `np.nanmean`, `np.nanstd`, `np.where`
- **SciPy** (already required): `scipy.stats.f`, `scipy.stats.chi2`, `scipy.stats.t`, `scipy.signal.savgol_filter`
- **Pandas** (already required): `pd.unique`

### Optional Dependencies NOT Recommended

- **scikit-learn**: Not adding as a required dependency
  - Would be useful for `StandardScaler` but adds complexity
  - NaN handling differs from PyPhi's requirements
  
- **factor_analyzer**: Not adding
  - Current varimax implementation is adequate
  - Would add unnecessary dependency

---

## Summary of Benefits

### Code Reduction
| Category | Lines Removed | Lines Added | Net Change |
|----------|--------------|-------------|------------|
| Statistical functions (`mean`, `std`) | ~30 | ~4 | -26 |
| F-distribution (`f95`, `f99`) | ~100 | ~10 | -90 |
| Chi-squared (`spe_ci`) | ~60 | ~15 | -45 |
| T-distribution (`single_score_conf_int`) | ~40 | ~10 | -30 |
| Savitzky-Golay | ~60 | ~20 | -40 |
| **Total** | **~290** | **~59** | **~-230** |

### Reliability Improvements
- Using battle-tested scipy.stats implementations
- Exact calculations vs interpolated lookup tables
- Better edge case handling

### Performance Improvements
- NumPy's nan-functions are C-optimized
- SciPy's implementations use optimized algorithms

---

## Implementation Priority

1. **Immediate** (Phase 1): Replace `mean()`, `std()`, `f95()`, `f99()`
2. **Phase 1**: Replace `spe_ci()`, `single_score_conf_int()`
3. **Phase 2**: Replace `spectra_savgol()`
4. **Phase 1**: Simplify `find()`, `unique()`
5. **Optional**: Simplify `meancenterscale()` using numpy nan-functions

---

## Testing Strategy

For each replacement:
1. Create comparison tests that verify old vs new produce same results (within tolerance)
2. Add edge case tests: empty arrays, all-NaN columns, single values
3. Add performance benchmarks for critical functions
4. Ensure backward compatibility of return types and shapes

