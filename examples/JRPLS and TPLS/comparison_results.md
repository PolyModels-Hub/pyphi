# JRPLS and TPLS Comparison Results

## New Refactored vs Legacy Implementation

**Date:** November 26, 2025  
**Test Script:** `jrpls_tpls_comparison_test.py`

---

## Summary

| Test                           | Status      | Notes                                              |
|:-------------------------------|:------------|:---------------------------------------------------|
| JRPLS (2 components)           | ⚡ CORE OK  | All core calculations match (diff ~1e-9 to 1e-11)  |
| JRPLS (4 components)           | ❌ FAILED   | Numerical drift accumulates with more LVs          |
| JRPLS Prediction (2-component) | ✓ PASSED    | Predictions match exactly                          |
| TPLS (2 components)            | ⚡ CORE OK  | All core calculations match (diff ~1e-9 to 1e-11)  |
| TPLS (4 components)            | ❌ FAILED   | Small numerical differences (~1e-3 to 1e-5)        |
| TPLS Prediction (2-component)  | ✓ PASSED    | Predictions match exactly                          |

**Core Calculations Passed: 4/6 tests**

---

## Detailed Results

### JRPLS (2 components) - ⚡ CORE OK

All core matrices match within numerical precision:

| Component           | Status  | Max Difference |
|:-------------------|:--------|:---------------|
| Scores (T)         | ✓ Match | 1.01e-10       |
| Y-Loadings (Q)     | ✓ Match | 2.93e-09       |
| U scores           | ✓ Match | 2.22e-10       |
| P (loadings)       | ✓ Match | ~1e-08         |
| S (weights)        | ✓ Match | ~1e-10         |
| H                  | ✓ Match | ~1e-09         |
| V                  | ✓ Match | ~1e-10         |
| Rscores            | ✓ Match | ~1e-09         |
| R2Y                | ✓ Match | 1.65e-11       |
| R2X per variable   | ✓ Match | 6.20e-10       |
| Hotelling T2       | ✓ Match | 4.86e-09       |
| SPE X              | ✓ Match | ~1e-10         |
| SPE R              | ✓ Match | ~1e-08         |
| SPE Y              | ✓ Match | 1.85e-09       |

### JRPLS (4 components) - ❌ FAILED

Significant numerical drift with 4 latent variables:

| Component                | Status       | Max Difference |
|:------------------------|:-------------|:---------------|
| Scores (T)              | ❌ Mismatch  | 2.21e-01       |
| Y-Loadings (Q)          | ❌ Mismatch  | 4.72e+00       |
| U scores                | ❌ Mismatch  | 1.17e+00       |
| P (loadings)            | ❌ Mismatch  | up to 1.85e+01 |
| Preprocessing (mx, sx, mr, sr) | ✓ Match | ~1e-15    |

**Note:** The preprocessing parameters match exactly, indicating the issue is in the iterative NIPALS algorithm convergence, not data handling.

### JRPLS Prediction (2-component model) - ✓ PASSED

| Component           | Status  | Max Difference |
|:-------------------|:--------|:---------------|
| New scores (Tnew)  | ✓ Match | 1.37e-09       |
| Predicted Y (Yhat) | ✓ Match | 9.57e-10       |
| SPE R (all mats)   | ✓ Match | ~1e-09         |

### TPLS (2 components) - ⚡ CORE OK

All core matrices match within numerical precision:

| Component           | Status  | Max Difference |
|:-------------------|:--------|:---------------|
| Scores (T)         | ✓ Match | 1.02e-08       |
| Y-Loadings (Q)     | ✓ Match | 1.26e-09       |
| U scores           | ✓ Match | 6.40e-08       |
| Wt (process wts)   | ✓ Match | 4.65e-10       |
| W (Z weights)      | ✓ Match | 2.17e-09       |
| Pz (Z loadings)    | ✓ Match | 1.68e-09       |
| P (loadings)       | ✓ Match | ~1e-09         |
| S (weights)        | ✓ Match | ~1e-09         |
| R2Y                | ✓ Match | 1.83e-11       |
| R2Z                | ✓ Match | 2.47e-11       |
| Hotelling T2       | ✓ Match | 5.55e-09       |
| SPE X              | ✓ Match | ~1e-09         |
| SPE R              | ✓ Match | ~1e-08         |
| SPE Y              | ✓ Match | 4.19e-09       |
| SPE Z              | ✓ Match | 9.68e-09       |

### TPLS (4 components) - ❌ FAILED

Small numerical differences (much smaller than JRPLS):

| Component                       | Status       | Max Difference |
|:-------------------------------|:-------------|:---------------|
| Scores (T)                     | ❌ Mismatch  | 3.24e-04       |
| Y-Loadings (Q)                 | ❌ Mismatch  | 1.29e-05       |
| U scores                       | ❌ Mismatch  | 2.20e-03       |
| Wt (process weights)           | ❌ Mismatch  | 2.69e-04       |
| Preprocessing (mx, sx, mr, sr, mz, sz) | ✓ Match | ~1e-15    |

**Note:** TPLS 4-component differences are very small (1e-3 to 1e-5) compared to JRPLS, suggesting the TPLS algorithm is more numerically stable.

### TPLS Prediction (2-component model) - ✓ PASSED

| Component           | Status  | Max Difference |
|:-------------------|:--------|:---------------|
| New scores (Tnew)  | ✓ Match | 5.64e-09       |
| Predicted Y (Yhat) | ✓ Match | 6.13e-10       |
| SPE Z              | ✓ Match | ~1e-09         |
| SPE R (all mats)   | ✓ Match | ~1e-09         |

---

## Statistical Limits

The new implementation uses **exact scipy calculations** instead of interpolated lookup tables:

- `f95/f99`: `scipy.stats.f.ppf()` instead of F-distribution table
- `spe_ci`: `scipy.stats.chi2.ppf()` instead of chi-squared table

This produces slightly more accurate values. Expected differences are 1-5%:

| Limit          | New (scipy) | Legacy (table) | Difference |
|:---------------|:------------|:--------------|:-----------|
| T2 95%         | 10.236      | 10.353        | 1.13%      |
| T2 99%         | 14.598      | 15.229        | 4.14%      |
| SPE Y 95%      | 13.788      | 13.733        | 0.40%      |
| SPE Y 99%      | 20.677      | 20.606        | 0.34%      |

---

## Key Findings

### ✓ What Works Correctly

1. **Basic algorithm is correct** - 2-component models match exactly between implementations
2. **Prediction code is correct** - When using matching models, predictions are identical
3. **Preprocessing is correct** - Mean/std calculations match exactly (diff ~1e-15)
4. **Statistical limits improved** - New scipy-based calculations are more accurate

### ⚠️ Areas Needing Investigation

1. **JRPLS 4-component divergence** - Larger numerical differences suggest potential:
   - Different convergence criteria
   - Numerical precision handling in NIPALS iterations
   - Possible algorithmic difference in later latent variables

2. **TPLS 4-component small drift** - Very small differences (1e-3 to 1e-5) are typical for iterative algorithms and may be acceptable

---

## How to Run the Comparison

```bash
cd /path/to/pyphi
python "examples/JRPLS and TPLS/jrpls_tpls_comparison_test.py"
```

---

## Dataset Information

- **File:** `jrpls_tpls_dataset.xlsx`
- **Materials:** MAT1, MAT2, MAT3, MAT4, MAT5
- **Quality variables:** 6 properties
- **Process variables:** 10 variables
- **Observations:** 105 blends (L001-L105)
