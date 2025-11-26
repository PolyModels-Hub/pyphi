#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PLS Comparison Script: New Refactored vs Legacy Implementation

This script compares the results from the new refactored PLS implementation
(src/pyphi/pls.py) against the legacy implementation (pyphi_legacy.py).

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
    """Compare two arrays and report differences, handling sign ambiguity for PLS.
    
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
    
    # Try sign-flipped comparison (common in PLS due to eigenvector sign ambiguity)
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


def run_pls_comparison(X: pd.DataFrame, Y: pd.DataFrame, n_components: int, 
                       test_name: str, **kwargs) -> dict:
    """Run PLS with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}, Components: {n_components}")
    print(f"Options: {kwargs}")
    
    # Run both implementations
    print("\nRunning legacy PLS...")
    pls_legacy = phi_legacy.pls(X, Y, n_components, shush=True, **kwargs)
    
    print("Running new PLS...")
    pls_new = phi_new.pls(X, Y, n_components, shush=True, **kwargs)
    
    # Compare results
    print("\n--- Comparing Results ---")
    results = {"test_name": test_name, "core_passed": True, "stat_limits_ok": True}
    
    # Core matrices (must match exactly)
    print("\nCore Matrices (must match exactly):")
    results["T"], _ = compare_arrays("Scores (T)", pls_new["T"], pls_legacy["T"])
    results["P"], _ = compare_arrays("X Loadings (P)", pls_new["P"], pls_legacy["P"])
    results["Q"], _ = compare_arrays("Y Loadings (Q)", pls_new["Q"], pls_legacy["Q"])
    results["W"], _ = compare_arrays("Weights (W)", pls_new["W"], pls_legacy["W"])
    results["Ws"], _ = compare_arrays("Modified Weights (Ws)", pls_new["Ws"], pls_legacy["Ws"])
    results["U"], _ = compare_arrays("Y Scores (U)", pls_new["U"], pls_legacy["U"])
    
    # Explained variance for X
    print("\nExplained Variance - X:")
    results["r2x"], _ = compare_arrays("R2X per component", pls_new["r2x"], pls_legacy["r2x"])
    results["r2xpv"], _ = compare_arrays("R2X per variable", pls_new["r2xpv"], pls_legacy["r2xpv"])
    
    # Explained variance for Y
    print("\nExplained Variance - Y:")
    results["r2y"], _ = compare_arrays("R2Y per component", pls_new["r2y"], pls_legacy["r2y"])
    results["r2ypv"], _ = compare_arrays("R2Y per variable", pls_new["r2ypv"], pls_legacy["r2ypv"])
    
    # Preprocessing parameters for X
    print("\nPreprocessing Parameters - X:")
    results["mx"], _ = compare_arrays("Mean X (mx)", pls_new["mx"], pls_legacy["mx"])
    results["sx"], _ = compare_arrays("Std X (sx)", pls_new["sx"], pls_legacy["sx"])
    
    # Preprocessing parameters for Y
    print("\nPreprocessing Parameters - Y:")
    results["my"], _ = compare_arrays("Mean Y (my)", pls_new["my"], pls_legacy["my"])
    results["sy"], _ = compare_arrays("Std Y (sy)", pls_new["sy"], pls_legacy["sy"])
    
    # Diagnostics - computed values (must match)
    print("\nDiagnostics - Computed Values (must match):")
    results["T2"], _ = compare_arrays("Hotelling T2", pls_new["T2"], pls_legacy["T2"])
    results["speX"], _ = compare_arrays("SPE X", pls_new["speX"], pls_legacy["speX"])
    results["speY"], _ = compare_arrays("SPE Y", pls_new["speY"], pls_legacy["speY"])
    
    # Diagnostics - statistical limits (expected to differ due to scipy vs lookup tables)
    print("\nDiagnostics - Statistical Limits (scipy vs interpolated tables - expected small diff):")
    results["T2_lim95"], _ = compare_scalars("T2 95% limit", pls_new["T2_lim95"], 
                                              pls_legacy["T2_lim95"], is_statistical_limit=True)
    results["T2_lim99"], _ = compare_scalars("T2 99% limit", pls_new["T2_lim99"], 
                                              pls_legacy["T2_lim99"], is_statistical_limit=True)
    results["speX_lim95"], _ = compare_scalars("SPE X 95% limit", pls_new["speX_lim95"], 
                                                pls_legacy["speX_lim95"], is_statistical_limit=True)
    results["speX_lim99"], _ = compare_scalars("SPE X 99% limit", pls_new["speX_lim99"], 
                                                pls_legacy["speX_lim99"], is_statistical_limit=True)
    results["speY_lim95"], _ = compare_scalars("SPE Y 95% limit", pls_new["speY_lim95"], 
                                                pls_legacy["speY_lim95"], is_statistical_limit=True)
    results["speY_lim99"], _ = compare_scalars("SPE Y 99% limit", pls_new["speY_lim99"], 
                                                pls_legacy["speY_lim99"], is_statistical_limit=True)
    
    # Cross-validation results (if applicable)
    if "q2Y" in pls_new and "q2Y" in pls_legacy:
        print("\nCross-Validation - Y:")
        results["q2Y"], _ = compare_arrays("Q2Y", pls_new["q2Y"], pls_legacy["q2Y"])
        results["q2Ypv"], _ = compare_arrays("Q2Y per variable", pls_new["q2Ypv"], pls_legacy["q2Ypv"])
        
    if "q2X" in pls_new and "q2X" in pls_legacy:
        print("\nCross-Validation - X:")
        results["q2X"], _ = compare_arrays("Q2X", pls_new["q2X"], pls_legacy["q2X"])
        results["q2Xpv"], _ = compare_arrays("Q2X per variable", pls_new["q2Xpv"], pls_legacy["q2Xpv"])
    
    # Determine pass/fail
    # Core results (non-statistical-limit fields)
    core_fields = ["T", "P", "Q", "W", "Ws", "U", 
                   "r2x", "r2xpv", "r2y", "r2ypv", 
                   "mx", "sx", "my", "sy", 
                   "T2", "speX", "speY"]
    if "q2Y" in results:
        core_fields.extend(["q2Y", "q2Ypv"])
    if "q2X" in results:
        core_fields.extend(["q2X", "q2Xpv"])
    results["core_passed"] = all(results.get(k, True) for k in core_fields)
    
    # Statistical limit fields
    stat_fields = ["T2_lim95", "T2_lim99", "speX_lim95", "speX_lim99", 
                   "speY_lim95", "speY_lim99"]
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


def run_pls_pred_comparison(X: pd.DataFrame, Y: pd.DataFrame, Xnew: np.ndarray, 
                            n_components: int, test_name: str) -> dict:
    """Run PLS prediction with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Build models
    pls_legacy = phi_legacy.pls(X, Y, n_components, shush=True)
    pls_new = phi_new.pls(X, Y, n_components, shush=True)
    
    # Run predictions
    print("Running legacy pls_pred...")
    pred_legacy = phi_legacy.pls_pred(Xnew, pls_legacy)
    
    print("Running new pls_pred...")
    pred_new = phi_new.pls_pred(Xnew, pls_new)
    
    # Compare results
    print("\n--- Comparing Prediction Results ---")
    results = {"test_name": test_name, "core_passed": True}
    
    results["Tnew"], _ = compare_arrays("New scores (Tnew)", pred_new["Tnew"], pred_legacy["Tnew"])
    results["Yhat"], _ = compare_arrays("Predicted Y (Yhat)", pred_new["Yhat"], pred_legacy["Yhat"])
    results["Xhat"], _ = compare_arrays("Reconstructed X (Xhat)", pred_new["Xhat"], pred_legacy["Xhat"])
    results["speX"], _ = compare_arrays("SPE X", pred_new["speX"], pred_legacy["speX"])
    results["T2"], _ = compare_arrays("Hotelling T2", pred_new["T2"], pred_legacy["T2"])
    
    results["core_passed"] = all(results.get(k, True) for k in ["Tnew", "Yhat", "Xhat", "speX", "T2"])
    results["passed"] = results["core_passed"]
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✓ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return results


def run_pls_cca_comparison(X: pd.DataFrame, Y: pd.DataFrame, n_components: int, 
                           test_name: str) -> dict:
    """Run PLS with CCA (OPLS-like) and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Build models with CCA
    print("\nRunning legacy PLS with CCA...")
    pls_legacy = phi_legacy.pls(X, Y, n_components, shush=True, cca=True)
    
    print("Running new PLS with CCA...")
    pls_new = phi_new.pls(X, Y, n_components, shush=True, cca=True)
    
    # Compare CCA-specific results
    print("\n--- Comparing CCA Results ---")
    results = {"test_name": test_name, "core_passed": True}
    
    # Core matrices
    results["T"], _ = compare_arrays("Scores (T)", pls_new["T"], pls_legacy["T"])
    results["P"], _ = compare_arrays("X Loadings (P)", pls_new["P"], pls_legacy["P"])
    
    # CCA-specific outputs
    if "Tcv" in pls_new and "Tcv" in pls_legacy:
        results["Tcv"], _ = compare_arrays("Covariant Scores (Tcv)", pls_new["Tcv"], pls_legacy["Tcv"])
    if "Pcv" in pls_new and "Pcv" in pls_legacy:
        results["Pcv"], _ = compare_arrays("Covariant Loadings (Pcv)", pls_new["Pcv"], pls_legacy["Pcv"])
    if "Wcv" in pls_new and "Wcv" in pls_legacy:
        results["Wcv"], _ = compare_arrays("Covariant Weights (Wcv)", pls_new["Wcv"], pls_legacy["Wcv"])
    if "Betacv" in pls_new and "Betacv" in pls_legacy:
        results["Betacv"], _ = compare_arrays("Covariant Beta (Betacv)", 
                                               np.atleast_1d(pls_new["Betacv"]), 
                                               np.atleast_1d(pls_legacy["Betacv"]))
    
    core_fields = ["T", "P"]
    if "Tcv" in results:
        core_fields.extend(["Tcv", "Pcv", "Wcv", "Betacv"])
    results["core_passed"] = all(results.get(k, True) for k in core_fields)
    results["passed"] = results["core_passed"]
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✓ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return results


def main():
    """Main comparison routine."""
    print("=" * 70)
    print("PLS COMPARISON: New Refactored vs Legacy Implementation")
    print("=" * 70)
    
    # Load data (same as Example_Script.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "Automobiles PLS.xls")
    
    print(f"\nLoading data from: {data_file}")
    Cars_Features = pd.read_excel(data_file, "Features", index_col=None, na_values=np.nan)
    Cars_Performance = pd.read_excel(data_file, "Performance", index_col=None, na_values=np.nan)
    print(f"X (Features) loaded: {Cars_Features.shape[0]} observations, {Cars_Features.shape[1]} columns")
    print(f"Y (Performance) loaded: {Cars_Performance.shape[0]} observations, {Cars_Performance.shape[1]} columns")
    print(f"X Columns: {list(Cars_Features.columns)}")
    print(f"Y Columns: {list(Cars_Performance.columns)}")
    
    all_results = []
    
    # ========================================================================
    # Test 1: Basic PLS with default options (mean-center + autoscale)
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 3,
        "Basic PLS (3 components, mcsX=True, mcsY=True)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 2: PLS with center only for X
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 3,
        "PLS with X center only",
        mcsX="center", mcsY=True
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 3: PLS with autoscale only for X
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 3,
        "PLS with X autoscale only",
        mcsX="autoscale", mcsY=True
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 4: PLS with no preprocessing for X
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 2,
        "PLS with no X preprocessing",
        mcsX=False, mcsY=True
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 5: PLS with cross-validation
    # ========================================================================
    print("\n" + "="*70)
    print("NOTE: Cross-validation uses random sampling, so results may differ slightly")
    print("="*70)
    
    # Set seed for reproducibility
    np.random.seed(42)
    pls_legacy_cv = phi_legacy.pls(Cars_Features, Cars_Performance, 3, cross_val=5, shush=True)
    
    np.random.seed(42)
    pls_new_cv = phi_new.pls(Cars_Features, Cars_Performance, 3, cross_val=5, shush=True)
    
    print(f"\n{'='*70}")
    print("TEST: PLS with cross-validation (5%)")
    print(f"{'='*70}")
    print("\n--- Comparing Cross-Validation Results ---")
    
    cv_results = {"test_name": "PLS with cross-validation", "core_passed": True}
    cv_results["T"], _ = compare_arrays("Scores (T)", pls_new_cv["T"], pls_legacy_cv["T"])
    cv_results["P"], _ = compare_arrays("Loadings (P)", pls_new_cv["P"], pls_legacy_cv["P"])
    cv_results["Q"], _ = compare_arrays("Y Loadings (Q)", pls_new_cv["Q"], pls_legacy_cv["Q"])
    cv_results["q2Y"], _ = compare_arrays("Q2Y", pls_new_cv["q2Y"], pls_legacy_cv["q2Y"])
    cv_results["core_passed"] = all(cv_results.get(k, True) for k in ["T", "P", "Q", "q2Y"])
    cv_results["passed"] = cv_results["core_passed"]
    
    print(f"\nTEST RESULT: {'✓ PASSED' if cv_results['passed'] else '❌ FAILED'}")
    all_results.append(cv_results)
    
    # ========================================================================
    # Test 6: PLS with cross-validation for both X and Y
    # ========================================================================
    np.random.seed(123)
    pls_legacy_cv_xy = phi_legacy.pls(Cars_Features, Cars_Performance, 3, 
                                       cross_val=5, cross_val_X=True, shush=True)
    
    np.random.seed(123)
    pls_new_cv_xy = phi_new.pls(Cars_Features, Cars_Performance, 3, 
                                 cross_val=5, cross_val_X=True, shush=True)
    
    print(f"\n{'='*70}")
    print("TEST: PLS with cross-validation for X and Y (5%)")
    print(f"{'='*70}")
    print("\n--- Comparing Cross-Validation Results ---")
    
    cv_xy_results = {"test_name": "PLS with cross-validation X+Y", "core_passed": True}
    cv_xy_results["T"], _ = compare_arrays("Scores (T)", pls_new_cv_xy["T"], pls_legacy_cv_xy["T"])
    cv_xy_results["q2Y"], _ = compare_arrays("Q2Y", pls_new_cv_xy["q2Y"], pls_legacy_cv_xy["q2Y"])
    cv_xy_results["q2X"], _ = compare_arrays("Q2X", pls_new_cv_xy["q2X"], pls_legacy_cv_xy["q2X"])
    cv_xy_results["core_passed"] = all(cv_xy_results.get(k, True) for k in ["T", "q2Y", "q2X"])
    cv_xy_results["passed"] = cv_xy_results["core_passed"]
    
    print(f"\nTEST RESULT: {'✓ PASSED' if cv_xy_results['passed'] else '❌ FAILED'}")
    all_results.append(cv_xy_results)
    
    # ========================================================================
    # Test 7: PLS Prediction
    # ========================================================================
    # Create test data (subset of original)
    X_array = np.array(Cars_Features.values[:, 1:]).astype(float)
    Xnew = X_array[:5, :]  # First 5 observations
    
    results = run_pls_pred_comparison(
        Cars_Features, Cars_Performance, Xnew, 3,
        "PLS Prediction (5 new observations)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 8: PLS with CCA (OPLS-like covariant space)
    # ========================================================================
    results = run_pls_cca_comparison(
        Cars_Features, Cars_Performance, 3,
        "PLS with CCA (OPLS-like)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 9: PLS with forced NIPALS (default behavior)
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 3,
        "PLS with forced NIPALS",
        force_nipals=True
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 10: PLS without forced NIPALS (uses SVD for complete data)
    # ========================================================================
    results = run_pls_comparison(
        Cars_Features, Cars_Performance, 3,
        "PLS without forced NIPALS (SVD)",
        force_nipals=False
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
        print("\n🎉 SUCCESS! All core PLS calculations match between implementations.")
        print("   The new refactored code produces equivalent results to the legacy version.")
        return 0
    else:
        print("\n⚠️  Some core tests failed. Please review the differences above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

