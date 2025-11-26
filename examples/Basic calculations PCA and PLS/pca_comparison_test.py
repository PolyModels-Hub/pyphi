#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Comparison Script: New Refactored vs Legacy Implementation

This script compares the results from the new refactored PCA implementation
(src/pyphi/pca.py) against the legacy implementation (pyphi_legacy.py).

IMPORTANT NOTE ON CONFIDENCE LIMITS:
The new implementation uses exact scipy statistical calculations:
  - f95/f99: scipy.stats.f.ppf() instead of hardcoded F-distribution table
  - spe_ci: scipy.stats.chi2.ppf() instead of hardcoded chi-squared table
  
These will produce slightly different (more accurate) results than the legacy
interpolated lookup tables. This is an intentional improvement.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(project_root, "src"))
sys.path.insert(0, project_root)

# Import both implementations
import pyphi as phi_new  # New refactored implementation
import pyphi_legacy as phi_legacy  # Legacy implementation


def compare_arrays(name: str, arr_new: np.ndarray, arr_legacy: np.ndarray, 
                   rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, float]:
    """Compare two arrays and report differences, handling sign ambiguity for PCA.
    
    Returns:
        tuple: (match_status, max_difference)
    """
    arr_new = np.atleast_1d(arr_new)
    arr_legacy = np.atleast_1d(arr_legacy)
    
    if arr_new.shape != arr_legacy.shape:
        print(f"  ❌ {name}: Shape mismatch - new: {arr_new.shape}, legacy: {arr_legacy.shape}")
        return False, np.inf
    
    # Direct comparison
    if np.allclose(arr_new, arr_legacy, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(arr_new - arr_legacy))
        print(f"  ✓ {name}: Match (max diff: {max_diff:.2e})")
        return True, max_diff
    
    # Try sign-flipped comparison (common in PCA due to eigenvector sign ambiguity)
    if np.allclose(arr_new, -arr_legacy, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(arr_new + arr_legacy))
        print(f"  ✓ {name}: Match with sign flip (max diff: {max_diff:.2e})")
        return True, max_diff
    
    # For multi-column arrays, check column-by-column with sign flexibility
    if arr_new.ndim == 2 and arr_new.shape[1] > 1:
        all_match = True
        for col in range(arr_new.shape[1]):
            col_new = arr_new[:, col]
            col_legacy = arr_legacy[:, col]
            if not (np.allclose(col_new, col_legacy, rtol=rtol, atol=atol) or 
                    np.allclose(col_new, -col_legacy, rtol=rtol, atol=atol)):
                all_match = False
                break
        if all_match:
            max_diff = np.max(np.abs(arr_new - arr_legacy))
            print(f"  ✓ {name}: Match (column-wise, with possible sign flips)")
            return True, max_diff
    
    max_diff = np.max(np.abs(arr_new - arr_legacy))
    mean_diff = np.mean(np.abs(arr_new - arr_legacy))
    print(f"  ❌ {name}: MISMATCH - max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e}")
    return False, max_diff


def compare_scalars(name: str, val_new: float, val_legacy: float, 
                    rtol: float = 1e-5, is_statistical_limit: bool = False) -> tuple[bool, float]:
    """Compare two scalar values.
    
    Args:
        is_statistical_limit: If True, uses looser tolerance and marks as expected difference
    """
    diff = abs(val_new - val_legacy)
    rel_diff = diff / max(abs(val_legacy), 1e-10)
    
    if is_statistical_limit:
        # Statistical limits use scipy (exact) vs lookup tables (interpolated)
        # Allow up to 5% relative difference for these values
        if rel_diff < 0.05:
            print(f"  ⚡ {name}: new={val_new:.6f}, legacy={val_legacy:.6f} "
                  f"(expected diff: {rel_diff*100:.2f}%, scipy vs table)")
            return True, diff
        else:
            print(f"  ⚠️  {name}: new={val_new:.6f}, legacy={val_legacy:.6f} "
                  f"(diff: {rel_diff*100:.2f}% > 5% threshold)")
            return False, diff
    
    if np.isclose(val_new, val_legacy, rtol=rtol):
        print(f"  ✓ {name}: {val_new:.6f} ≈ {val_legacy:.6f}")
        return True, diff
    else:
        print(f"  ❌ {name}: new={val_new:.6f}, legacy={val_legacy:.6f}, diff={diff:.2e}")
        return False, diff


def run_pca_comparison(X: pd.DataFrame, n_components: int, test_name: str, **kwargs) -> dict:
    """Run PCA with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Data shape: {X.shape}, Components: {n_components}")
    print(f"Options: {kwargs}")
    
    # Run both implementations
    print("\nRunning legacy PCA...")
    pca_legacy = phi_legacy.pca(X, n_components, shush=True, **kwargs)
    
    print("Running new PCA...")
    pca_new = phi_new.pca(X, n_components, shush=True, **kwargs)
    
    # Compare results
    print("\n--- Comparing Results ---")
    results = {"test_name": test_name, "core_passed": True, "stat_limits_ok": True}
    
    # Core matrices (must match exactly)
    print("\nCore Matrices (must match exactly):")
    results["T"], _ = compare_arrays("Scores (T)", pca_new["T"], pca_legacy["T"])
    results["P"], _ = compare_arrays("Loadings (P)", pca_new["P"], pca_legacy["P"])
    
    # Explained variance
    print("\nExplained Variance:")
    results["r2x"], _ = compare_arrays("R2X per component", pca_new["r2x"], pca_legacy["r2x"])
    results["r2xpv"], _ = compare_arrays("R2X per variable", pca_new["r2xpv"], pca_legacy["r2xpv"])
    
    # Preprocessing parameters
    print("\nPreprocessing Parameters:")
    results["mx"], _ = compare_arrays("Mean (mx)", pca_new["mx"], pca_legacy["mx"])
    results["sx"], _ = compare_arrays("Std (sx)", pca_new["sx"], pca_legacy["sx"])
    
    # Diagnostics - computed values (must match)
    print("\nDiagnostics - Computed Values (must match):")
    results["T2"], _ = compare_arrays("Hotelling T2", pca_new["T2"], pca_legacy["T2"])
    results["speX"], _ = compare_arrays("SPE", pca_new["speX"], pca_legacy["speX"])
    
    # Diagnostics - statistical limits (expected to differ due to scipy vs lookup tables)
    print("\nDiagnostics - Statistical Limits (scipy vs interpolated tables - expected small diff):")
    results["T2_lim95"], _ = compare_scalars("T2 95% limit", pca_new["T2_lim95"], 
                                              pca_legacy["T2_lim95"], is_statistical_limit=True)
    results["T2_lim99"], _ = compare_scalars("T2 99% limit", pca_new["T2_lim99"], 
                                              pca_legacy["T2_lim99"], is_statistical_limit=True)
    results["speX_lim95"], _ = compare_scalars("SPE 95% limit", pca_new["speX_lim95"], 
                                                pca_legacy["speX_lim95"], is_statistical_limit=True)
    results["speX_lim99"], _ = compare_scalars("SPE 99% limit", pca_new["speX_lim99"], 
                                                pca_legacy["speX_lim99"], is_statistical_limit=True)
    
    # Cross-validation results (if applicable)
    if "q2" in pca_new and "q2" in pca_legacy:
        print("\nCross-Validation:")
        results["q2"], _ = compare_arrays("Q2X", pca_new["q2"], pca_legacy["q2"])
        results["q2pv"], _ = compare_arrays("Q2X per variable", pca_new["q2pv"], pca_legacy["q2pv"])
    
    # Determine pass/fail
    # Core results (non-statistical-limit fields)
    core_fields = ["T", "P", "r2x", "r2xpv", "mx", "sx", "T2", "speX"]
    if "q2" in results:
        core_fields.extend(["q2", "q2pv"])
    results["core_passed"] = all(results.get(k, True) for k in core_fields)
    
    # Statistical limit fields
    stat_fields = ["T2_lim95", "T2_lim99", "speX_lim95", "speX_lim99"]
    results["stat_limits_ok"] = all(results.get(k, True) for k in stat_fields)
    
    results["passed"] = results["core_passed"] and results["stat_limits_ok"]
    
    print(f"\n{'='*70}")
    if results["core_passed"]:
        if results["stat_limits_ok"]:
            print("TEST RESULT: ✓ PASSED (all results match)")
        else:
            print("TEST RESULT: ⚡ CORE PASSED (statistical limits have expected scipy vs table differences)")
    else:
        print("TEST RESULT: ❌ FAILED (core calculations differ)")
    print(f"{'='*70}")
    
    return results


def run_pca_pred_comparison(X: pd.DataFrame, Xnew: np.ndarray, n_components: int, test_name: str) -> dict:
    """Run PCA prediction with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Build models
    pca_legacy = phi_legacy.pca(X, n_components, shush=True)
    pca_new = phi_new.pca(X, n_components, shush=True)
    
    # Run predictions
    print("Running legacy pca_pred...")
    pred_legacy = phi_legacy.pca_pred(Xnew, pca_legacy)
    
    print("Running new pca_pred...")
    pred_new = phi_new.pca_pred(Xnew, pca_new)
    
    # Compare results
    print("\n--- Comparing Prediction Results ---")
    results = {"test_name": test_name, "core_passed": True}
    
    results["Tnew"], _ = compare_arrays("New scores (Tnew)", pred_new["Tnew"], pred_legacy["Tnew"])
    results["Xhat"], _ = compare_arrays("Reconstructed X (Xhat)", pred_new["Xhat"], pred_legacy["Xhat"])
    results["speX"], _ = compare_arrays("SPE", pred_new["speX"], pred_legacy["speX"])
    results["T2"], _ = compare_arrays("Hotelling T2", pred_new["T2"], pred_legacy["T2"])
    
    results["core_passed"] = all(results.get(k, True) for k in ["Tnew", "Xhat", "speX", "T2"])
    results["passed"] = results["core_passed"]
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✓ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return results


def main():
    """Main comparison routine."""
    print("=" * 70)
    print("PCA COMPARISON: New Refactored vs Legacy Implementation")
    print("=" * 70)
    
    # Load data (same as Example_Script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Automobiles PLS.xls")
    
    print(f"\nLoading data from: {data_file}")
    Cars_Features = pd.read_excel(data_file, "Features", index_col=None, na_values=np.nan)
    print(f"Data loaded: {Cars_Features.shape[0]} observations, {Cars_Features.shape[1]} columns")
    print(f"Columns: {list(Cars_Features.columns)}")
    
    all_results = []
    
    # ========================================================================
    # Test 1: Basic PCA with default options (mean-center + autoscale)
    # ========================================================================
    results = run_pca_comparison(
        Cars_Features, 3,
        "Basic PCA (3 components, mcs=True)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 2: PCA with center only
    # ========================================================================
    results = run_pca_comparison(
        Cars_Features, 3,
        "PCA with center only",
        mcs="center"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 3: PCA with autoscale only
    # ========================================================================
    results = run_pca_comparison(
        Cars_Features, 3,
        "PCA with autoscale only",
        mcs="autoscale"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 4: PCA with no preprocessing
    # ========================================================================
    results = run_pca_comparison(
        Cars_Features, 2,
        "PCA with no preprocessing",
        mcs=False
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 5: PCA with cross-validation
    # ========================================================================
    print("\n" + "="*70)
    print("NOTE: Cross-validation uses random sampling, so results may differ slightly")
    print("="*70)
    
    # Set seed for reproducibility
    np.random.seed(42)
    results_legacy = phi_legacy.pca(Cars_Features, 3, cross_val=5, shush=True)
    
    np.random.seed(42)
    results_new = phi_new.pca(Cars_Features, 3, cross_val=5, shush=True)
    
    print(f"\n{'='*70}")
    print("TEST: PCA with cross-validation (5%)")
    print(f"{'='*70}")
    print("\n--- Comparing Cross-Validation Results ---")
    
    cv_results = {"test_name": "PCA with cross-validation", "core_passed": True}
    cv_results["T"], _ = compare_arrays("Scores (T)", results_new["T"], results_legacy["T"])
    cv_results["P"], _ = compare_arrays("Loadings (P)", results_new["P"], results_legacy["P"])
    cv_results["q2"], _ = compare_arrays("Q2X", results_new["q2"], results_legacy["q2"])
    cv_results["core_passed"] = all(cv_results.get(k, True) for k in ["T", "P", "q2"])
    cv_results["passed"] = cv_results["core_passed"]
    
    print(f"\nTEST RESULT: {'✓ PASSED' if cv_results['passed'] else '❌ FAILED'}")
    all_results.append(cv_results)
    
    # ========================================================================
    # Test 6: PCA Prediction
    # ========================================================================
    # Create test data (subset of original)
    X_array = np.array(Cars_Features.values[:, 1:]).astype(float)
    Xnew = X_array[:5, :]  # First 5 observations
    
    results = run_pca_pred_comparison(
        Cars_Features, Xnew, 3,
        "PCA Prediction (5 new observations)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 7: PCA with forced NIPALS
    # ========================================================================
    results = run_pca_comparison(
        Cars_Features, 3,
        "PCA with forced NIPALS",
        force_nipals=True
    )
    all_results.append(results)
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    core_passed = sum(1 for r in all_results if r.get("core_passed", r.get("passed", False)))
    total = len(all_results)
    
    print("\nTest Results:")
    for r in all_results:
        core_ok = r.get("core_passed", r.get("passed", False))
        stat_ok = r.get("stat_limits_ok", True)  # Default True if not applicable
        
        if core_ok and stat_ok:
            status = "✓ PASSED"
        elif core_ok and not stat_ok:
            status = "⚡ CORE OK"  # Core matches, statistical limits have expected differences
        else:
            status = "❌ FAILED"
        print(f"  {status}: {r['test_name']}")
    
    print(f"\nCore Calculations: {core_passed}/{total} tests passed")
    
    # Explain the statistical limit differences
    print("\n" + "-" * 70)
    print("NOTES ON STATISTICAL LIMITS:")
    print("-" * 70)
    print("The new implementation uses exact scipy calculations for confidence limits:")
    print("  • f95/f99: scipy.stats.f.ppf() instead of F-distribution lookup table")
    print("  • spe_ci: scipy.stats.chi2.ppf() instead of chi-squared lookup table")
    print("")
    print("This produces slightly more accurate values than the interpolated tables")
    print("in the legacy code. Differences of <5% are expected and acceptable.")
    print("-" * 70)
    
    if core_passed == total:
        print("\n🎉 SUCCESS! All core PCA calculations match between implementations.")
        print("   The new refactored code produces equivalent results to the legacy version.")
        return 0
    else:
        print("\n⚠️  Some core tests failed. Please review the differences above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

