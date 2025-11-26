#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JRPLS and TPLS Comparison Script: New Refactored vs Legacy Implementation

This script compares the results from the new refactored JRPLS/TPLS implementations
(src/pyphi/advanced_pls.py) against the legacy implementation (pyphi_legacy.py).

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
    """Compare two arrays and report differences, handling sign ambiguity.
    
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


def compare_array_lists(name: str, list_new: list, list_legacy: list,
                        rtol: float = 1e-5, atol: float = 1e-8) -> tuple[bool, float]:
    """Compare two lists of arrays (e.g., per-material arrays).
    
    Returns:
        tuple: (match_status, max_difference)
    """
    if len(list_new) != len(list_legacy):
        print(f"  ❌ {name}: List length mismatch - new: {len(list_new)}, legacy: {len(list_legacy)}")
        return False, np.inf
    
    all_match = True
    max_diff_overall = 0.0
    
    for i, (arr_new, arr_legacy) in enumerate(zip(list_new, list_legacy)):
        match, diff = compare_arrays(f"{name}[{i}]", arr_new, arr_legacy, rtol, atol)
        if not match:
            all_match = False
        max_diff_overall = max(max_diff_overall, diff)
    
    return all_match, max_diff_overall


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


def compare_scalar_lists(name: str, list_new: list, list_legacy: list,
                         is_statistical_limit: bool = False) -> tuple[bool, float]:
    """Compare two lists of scalars.
    
    Returns:
        tuple: (match_status, max_difference)
    """
    if len(list_new) != len(list_legacy):
        print(f"  ❌ {name}: List length mismatch - new: {len(list_new)}, legacy: {len(list_legacy)}")
        return False, np.inf
    
    all_match = True
    max_diff_overall = 0.0
    
    for i, (val_new, val_legacy) in enumerate(zip(list_new, list_legacy)):
        match, diff = compare_scalars(f"{name}[{i}]", val_new, val_legacy, 
                                       is_statistical_limit=is_statistical_limit)
        if not match:
            all_match = False
        max_diff_overall = max(max_diff_overall, diff)
    
    return all_match, max_diff_overall


def load_data():
    """Load the JRPLS/TPLS dataset using both implementations."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "jrpls_tpls_dataset.xlsx")
    
    print(f"\nLoading data from: {data_file}")
    
    # Use legacy implementation for data loading (they should be identical)
    jr, materials = phi_legacy.parse_materials(data_file, 'Materials')
    x = []
    for m in materials:
        x_ = pd.read_excel(data_file, sheet_name=m)
        x.append(x_)
    
    xc, jrc = phi_legacy.reconcile_rows_to_columns(x, jr)
    
    quality = pd.read_excel(data_file, sheet_name='QUALITY')
    process = pd.read_excel(data_file, sheet_name='PROCESS')
    
    jrc.append(process)
    jrc.append(quality)
    AUX = phi_legacy.reconcile_rows(jrc)
    
    JR_ = AUX[:-2]
    process = AUX[-2]
    quality = AUX[-1]
    
    # Build dictionaries
    Ri = {}
    for j, m in zip(JR_, materials):
        Ri[m] = j
    Xi = {}
    for x_, m in zip(xc, materials):
        Xi[m] = x_
    
    print(f"Materials: {materials}")
    print(f"Quality shape: {quality.shape}")
    print(f"Process shape: {process.shape}")
    
    return Xi, Ri, quality, process, materials


def run_jrpls_comparison(Xi: dict, Ri: dict, Y: pd.DataFrame, n_components: int, 
                          test_name: str) -> dict:
    """Run JRPLS with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Components: {n_components}")
    
    # Run both implementations
    print("\nRunning legacy JRPLS...")
    jrpls_legacy = phi_legacy.jrpls(Xi, Ri, Y, n_components, shush=True)
    
    print("Running new JRPLS...")
    jrpls_new = phi_new.jrpls(Xi, Ri, Y, n_components, shush=True)
    
    # Compare results
    print("\n--- Comparing Results ---")
    results = {"test_name": test_name, "core_passed": True, "stat_limits_ok": True}
    
    # Core matrices (must match exactly)
    print("\nCore Matrices (must match exactly):")
    results["T"], _ = compare_arrays("Scores (T)", jrpls_new["T"], jrpls_legacy["T"])
    results["Q"], _ = compare_arrays("Y-Loadings (Q)", jrpls_new["Q"], jrpls_legacy["Q"])
    results["U"], _ = compare_arrays("U scores", jrpls_new["U"], jrpls_legacy["U"])
    
    # Per-material matrices
    print("\nPer-Material Matrices:")
    results["P"], _ = compare_array_lists("P (loadings)", jrpls_new["P"], jrpls_legacy["P"])
    results["S"], _ = compare_array_lists("S (weights)", jrpls_new["S"], jrpls_legacy["S"])
    results["H"], _ = compare_array_lists("H", jrpls_new["H"], jrpls_legacy["H"])
    results["V"], _ = compare_array_lists("V", jrpls_new["V"], jrpls_legacy["V"])
    results["Rscores"], _ = compare_array_lists("Rscores", jrpls_new["Rscores"], jrpls_legacy["Rscores"])
    
    # Explained variance
    print("\nExplained Variance:")
    results["r2y"], _ = compare_arrays("R2Y", jrpls_new["r2y"], jrpls_legacy["r2y"])
    results["r2ypv"], _ = compare_arrays("R2Y per variable", jrpls_new["r2ypv"], jrpls_legacy["r2ypv"])
    results["r2xpv"], _ = compare_arrays("R2X per variable (all)", jrpls_new["r2xpv"], jrpls_legacy["r2xpv"])
    
    # Per-material R2
    print("\nPer-Material R2:")
    results["r2xi"], _ = compare_array_lists("r2xi", jrpls_new["r2xi"], jrpls_legacy["r2xi"])
    results["r2ri"], _ = compare_array_lists("r2ri", jrpls_new["r2ri"], jrpls_legacy["r2ri"])
    
    # Preprocessing parameters
    print("\nPreprocessing Parameters:")
    results["mx"], _ = compare_array_lists("Mean X (mx)", jrpls_new["mx"], jrpls_legacy["mx"])
    results["sx"], _ = compare_array_lists("Std X (sx)", jrpls_new["sx"], jrpls_legacy["sx"])
    results["mr"], _ = compare_array_lists("Mean R (mr)", jrpls_new["mr"], jrpls_legacy["mr"])
    results["sr"], _ = compare_array_lists("Std R (sr)", jrpls_new["sr"], jrpls_legacy["sr"])
    results["my"], _ = compare_arrays("Mean Y (my)", jrpls_new["my"], jrpls_legacy["my"])
    results["sy"], _ = compare_arrays("Std Y (sy)", jrpls_new["sy"], jrpls_legacy["sy"])
    
    # Diagnostics - computed values (must match)
    print("\nDiagnostics - Computed Values (must match):")
    results["T2"], _ = compare_arrays("Hotelling T2", jrpls_new["T2"], jrpls_legacy["T2"])
    results["speX"], _ = compare_array_lists("SPE X", jrpls_new["speX"], jrpls_legacy["speX"])
    results["speR"], _ = compare_array_lists("SPE R", jrpls_new["speR"], jrpls_legacy["speR"])
    results["speY"], _ = compare_arrays("SPE Y", jrpls_new["speY"], jrpls_legacy["speY"])
    
    # Diagnostics - statistical limits (expected to differ due to scipy vs lookup tables)
    print("\nDiagnostics - Statistical Limits (scipy vs interpolated tables - expected small diff):")
    results["T2_lim95"], _ = compare_scalars("T2 95% limit", jrpls_new["T2_lim95"], 
                                              jrpls_legacy["T2_lim95"], is_statistical_limit=True)
    results["T2_lim99"], _ = compare_scalars("T2 99% limit", jrpls_new["T2_lim99"], 
                                              jrpls_legacy["T2_lim99"], is_statistical_limit=True)
    results["speY_lim95"], _ = compare_scalars("SPE Y 95% limit", jrpls_new["speY_lim95"], 
                                                jrpls_legacy["speY_lim95"], is_statistical_limit=True)
    results["speY_lim99"], _ = compare_scalars("SPE Y 99% limit", jrpls_new["speY_lim99"], 
                                                jrpls_legacy["speY_lim99"], is_statistical_limit=True)
    
    # Per-material SPE limits
    print("\nPer-Material SPE Limits:")
    results["speX_lim95"], _ = compare_scalar_lists("SPE X 95% limits", 
                                                     jrpls_new["speX_lim95"], jrpls_legacy["speX_lim95"],
                                                     is_statistical_limit=True)
    results["speX_lim99"], _ = compare_scalar_lists("SPE X 99% limits",
                                                     jrpls_new["speX_lim99"], jrpls_legacy["speX_lim99"],
                                                     is_statistical_limit=True)
    results["speR_lim95"], _ = compare_scalar_lists("SPE R 95% limits",
                                                     jrpls_new["speR_lim95"], jrpls_legacy["speR_lim95"],
                                                     is_statistical_limit=True)
    results["speR_lim99"], _ = compare_scalar_lists("SPE R 99% limits",
                                                     jrpls_new["speR_lim99"], jrpls_legacy["speR_lim99"],
                                                     is_statistical_limit=True)
    
    # Determine pass/fail
    # Core results (non-statistical-limit fields)
    core_fields = ["T", "Q", "U", "P", "S", "H", "V", "Rscores",
                   "r2y", "r2ypv", "r2xpv", "r2xi", "r2ri",
                   "mx", "sx", "mr", "sr", "my", "sy",
                   "T2", "speX", "speR", "speY"]
    results["core_passed"] = all(results.get(k, True) for k in core_fields)
    
    # Statistical limit fields
    stat_fields = ["T2_lim95", "T2_lim99", "speY_lim95", "speY_lim99",
                   "speX_lim95", "speX_lim99", "speR_lim95", "speR_lim99"]
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
    
    return results, jrpls_new, jrpls_legacy


def run_jrpls_pred_comparison(jrpls_new: dict, jrpls_legacy: dict, rnew: dict, 
                               test_name: str) -> dict:
    """Run JRPLS prediction with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Run predictions
    print("Running legacy jrpls_pred...")
    pred_legacy = phi_legacy.jrpls_pred(rnew, jrpls_legacy)
    
    print("Running new jrpls_pred...")
    pred_new = phi_new.jrpls_pred(rnew, jrpls_new)
    
    # Compare results
    print("\n--- Comparing Prediction Results ---")
    results = {"test_name": test_name, "core_passed": True}
    
    results["Tnew"], _ = compare_arrays("New scores (Tnew)", pred_new["Tnew"], pred_legacy["Tnew"])
    results["Yhat"], _ = compare_arrays("Predicted Y (Yhat)", pred_new["Yhat"], pred_legacy["Yhat"])
    
    # speR is a list of scalars
    print("\nPer-Material SPE R:")
    all_match = True
    for i, (spe_new, spe_legacy) in enumerate(zip(pred_new["speR"], pred_legacy["speR"])):
        match, _ = compare_scalars(f"speR[{i}]", spe_new, spe_legacy)
        if not match:
            all_match = False
    results["speR"] = all_match
    
    results["core_passed"] = all(results.get(k, True) for k in ["Tnew", "Yhat", "speR"])
    results["passed"] = results["core_passed"]
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✓ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return results


def run_tpls_comparison(Xi: dict, Ri: dict, Z: pd.DataFrame, Y: pd.DataFrame, 
                         n_components: int, test_name: str) -> dict:
    """Run TPLS with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    print(f"Components: {n_components}")
    
    # Run both implementations
    print("\nRunning legacy TPLS...")
    tpls_legacy = phi_legacy.tpls(Xi, Ri, Z, Y, n_components, shush=True)
    
    print("Running new TPLS...")
    tpls_new = phi_new.tpls(Xi, Ri, Z, Y, n_components, shush=True)
    
    # Compare results
    print("\n--- Comparing Results ---")
    results = {"test_name": test_name, "core_passed": True, "stat_limits_ok": True}
    
    # Core matrices (must match exactly)
    print("\nCore Matrices (must match exactly):")
    results["T"], _ = compare_arrays("Scores (T)", tpls_new["T"], tpls_legacy["T"])
    results["Q"], _ = compare_arrays("Y-Loadings (Q)", tpls_new["Q"], tpls_legacy["Q"])
    results["U"], _ = compare_arrays("U scores", tpls_new["U"], tpls_legacy["U"])
    results["Wt"], _ = compare_arrays("Wt (process weights)", tpls_new["Wt"], tpls_legacy["Wt"])
    results["W"], _ = compare_arrays("W (Z weights)", tpls_new["W"], tpls_legacy["W"])
    results["Pz"], _ = compare_arrays("Pz (Z loadings)", tpls_new["Pz"], tpls_legacy["Pz"])
    
    # Per-material matrices
    print("\nPer-Material Matrices:")
    results["P"], _ = compare_array_lists("P (loadings)", tpls_new["P"], tpls_legacy["P"])
    results["S"], _ = compare_array_lists("S (weights)", tpls_new["S"], tpls_legacy["S"])
    results["H"], _ = compare_array_lists("H", tpls_new["H"], tpls_legacy["H"])
    results["V"], _ = compare_array_lists("V", tpls_new["V"], tpls_legacy["V"])
    results["Rscores"], _ = compare_array_lists("Rscores", tpls_new["Rscores"], tpls_legacy["Rscores"])
    
    # Explained variance
    print("\nExplained Variance:")
    results["r2y"], _ = compare_arrays("R2Y", tpls_new["r2y"], tpls_legacy["r2y"])
    results["r2ypv"], _ = compare_arrays("R2Y per variable", tpls_new["r2ypv"], tpls_legacy["r2ypv"])
    results["r2z"], _ = compare_arrays("R2Z", tpls_new["r2z"], tpls_legacy["r2z"])
    results["r2zpv"], _ = compare_arrays("R2Z per variable", tpls_new["r2zpv"], tpls_legacy["r2zpv"])
    results["r2xpv"], _ = compare_arrays("R2X per variable (all)", tpls_new["r2xpv"], tpls_legacy["r2xpv"])
    
    # Per-material R2
    print("\nPer-Material R2:")
    results["r2xi"], _ = compare_array_lists("r2xi", tpls_new["r2xi"], tpls_legacy["r2xi"])
    results["r2ri"], _ = compare_array_lists("r2ri", tpls_new["r2ri"], tpls_legacy["r2ri"])
    
    # Preprocessing parameters
    print("\nPreprocessing Parameters:")
    results["mx"], _ = compare_array_lists("Mean X (mx)", tpls_new["mx"], tpls_legacy["mx"])
    results["sx"], _ = compare_array_lists("Std X (sx)", tpls_new["sx"], tpls_legacy["sx"])
    results["mr"], _ = compare_array_lists("Mean R (mr)", tpls_new["mr"], tpls_legacy["mr"])
    results["sr"], _ = compare_array_lists("Std R (sr)", tpls_new["sr"], tpls_legacy["sr"])
    results["my"], _ = compare_arrays("Mean Y (my)", tpls_new["my"], tpls_legacy["my"])
    results["sy"], _ = compare_arrays("Std Y (sy)", tpls_new["sy"], tpls_legacy["sy"])
    results["mz"], _ = compare_arrays("Mean Z (mz)", tpls_new["mz"], tpls_legacy["mz"])
    results["sz"], _ = compare_arrays("Std Z (sz)", tpls_new["sz"], tpls_legacy["sz"])
    
    # Diagnostics - computed values (must match)
    print("\nDiagnostics - Computed Values (must match):")
    results["T2"], _ = compare_arrays("Hotelling T2", tpls_new["T2"], tpls_legacy["T2"])
    results["speX"], _ = compare_array_lists("SPE X", tpls_new["speX"], tpls_legacy["speX"])
    results["speR"], _ = compare_array_lists("SPE R", tpls_new["speR"], tpls_legacy["speR"])
    results["speY"], _ = compare_arrays("SPE Y", tpls_new["speY"], tpls_legacy["speY"])
    results["speZ"], _ = compare_arrays("SPE Z", tpls_new["speZ"], tpls_legacy["speZ"])
    
    # Diagnostics - statistical limits (expected to differ due to scipy vs lookup tables)
    print("\nDiagnostics - Statistical Limits (scipy vs interpolated tables - expected small diff):")
    results["T2_lim95"], _ = compare_scalars("T2 95% limit", tpls_new["T2_lim95"], 
                                              tpls_legacy["T2_lim95"], is_statistical_limit=True)
    results["T2_lim99"], _ = compare_scalars("T2 99% limit", tpls_new["T2_lim99"], 
                                              tpls_legacy["T2_lim99"], is_statistical_limit=True)
    results["speY_lim95"], _ = compare_scalars("SPE Y 95% limit", tpls_new["speY_lim95"], 
                                                tpls_legacy["speY_lim95"], is_statistical_limit=True)
    results["speY_lim99"], _ = compare_scalars("SPE Y 99% limit", tpls_new["speY_lim99"], 
                                                tpls_legacy["speY_lim99"], is_statistical_limit=True)
    results["speZ_lim95"], _ = compare_scalars("SPE Z 95% limit", tpls_new["speZ_lim95"], 
                                                tpls_legacy["speZ_lim95"], is_statistical_limit=True)
    results["speZ_lim99"], _ = compare_scalars("SPE Z 99% limit", tpls_new["speZ_lim99"], 
                                                tpls_legacy["speZ_lim99"], is_statistical_limit=True)
    
    # Per-material SPE limits
    print("\nPer-Material SPE Limits:")
    results["speX_lim95"], _ = compare_scalar_lists("SPE X 95% limits", 
                                                     tpls_new["speX_lim95"], tpls_legacy["speX_lim95"],
                                                     is_statistical_limit=True)
    results["speX_lim99"], _ = compare_scalar_lists("SPE X 99% limits",
                                                     tpls_new["speX_lim99"], tpls_legacy["speX_lim99"],
                                                     is_statistical_limit=True)
    results["speR_lim95"], _ = compare_scalar_lists("SPE R 95% limits",
                                                     tpls_new["speR_lim95"], tpls_legacy["speR_lim95"],
                                                     is_statistical_limit=True)
    results["speR_lim99"], _ = compare_scalar_lists("SPE R 99% limits",
                                                     tpls_new["speR_lim99"], tpls_legacy["speR_lim99"],
                                                     is_statistical_limit=True)
    
    # Determine pass/fail
    # Core results (non-statistical-limit fields)
    core_fields = ["T", "Q", "U", "Wt", "W", "Pz", "P", "S", "H", "V", "Rscores",
                   "r2y", "r2ypv", "r2z", "r2zpv", "r2xpv", "r2xi", "r2ri",
                   "mx", "sx", "mr", "sr", "my", "sy", "mz", "sz",
                   "T2", "speX", "speR", "speY", "speZ"]
    results["core_passed"] = all(results.get(k, True) for k in core_fields)
    
    # Statistical limit fields
    stat_fields = ["T2_lim95", "T2_lim99", "speY_lim95", "speY_lim99", "speZ_lim95", "speZ_lim99",
                   "speX_lim95", "speX_lim99", "speR_lim95", "speR_lim99"]
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
    
    return results, tpls_new, tpls_legacy


def run_tpls_pred_comparison(tpls_new: dict, tpls_legacy: dict, rnew: dict, 
                              znew: np.ndarray, test_name: str) -> dict:
    """Run TPLS prediction with both implementations and compare results."""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Run predictions
    print("Running legacy tpls_pred...")
    pred_legacy = phi_legacy.tpls_pred(rnew, znew, tpls_legacy)
    
    print("Running new tpls_pred...")
    pred_new = phi_new.tpls_pred(rnew, znew, tpls_new)
    
    # Compare results
    print("\n--- Comparing Prediction Results ---")
    results = {"test_name": test_name, "core_passed": True}
    
    results["Tnew"], _ = compare_arrays("New scores (Tnew)", pred_new["Tnew"], pred_legacy["Tnew"])
    results["Yhat"], _ = compare_arrays("Predicted Y (Yhat)", pred_new["Yhat"], pred_legacy["Yhat"])
    results["speZ"], _ = compare_scalars("SPE Z", pred_new["speZ"], pred_legacy["speZ"])
    
    # speR is a list of scalars
    print("\nPer-Material SPE R:")
    all_match = True
    for i, (spe_new, spe_legacy) in enumerate(zip(pred_new["speR"], pred_legacy["speR"])):
        match, _ = compare_scalars(f"speR[{i}]", spe_new, spe_legacy)
        if not match:
            all_match = False
    results["speR"] = all_match
    
    results["core_passed"] = all(results.get(k, True) for k in ["Tnew", "Yhat", "speZ", "speR"])
    results["passed"] = results["core_passed"]
    
    print(f"\n{'='*70}")
    print(f"TEST RESULT: {'✓ PASSED' if results['passed'] else '❌ FAILED'}")
    print(f"{'='*70}")
    
    return results


def main():
    """Main comparison routine."""
    print("=" * 70)
    print("JRPLS/TPLS COMPARISON: New Refactored vs Legacy Implementation")
    print("=" * 70)
    
    # Load data
    Xi, Ri, quality, process, materials = load_data()
    
    all_results = []
    
    # ========================================================================
    # Test 1: JRPLS with 2 components (stable, should match exactly)
    # ========================================================================
    results, jrpls_new_2, jrpls_legacy_2 = run_jrpls_comparison(
        Xi, Ri, quality, 2,
        "JRPLS (2 components)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 2: JRPLS with 4 components (may show numerical drift)
    # ========================================================================
    results, jrpls_new_4, jrpls_legacy_4 = run_jrpls_comparison(
        Xi, Ri, quality, 4,
        "JRPLS (4 components)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 3: JRPLS Prediction (using 2-component model for fair comparison)
    # ========================================================================
    rnew = {
        'MAT1': [('A0129', 0.557949425), ('A0130', 0.442050575)],
        'MAT2': [('Lac0003', 1)],
        'MAT3': [('TLC018', 1)],
        'MAT4': [('M0012', 1)],
        'MAT5': [('CS0017', 1)]
    }
    results = run_jrpls_pred_comparison(
        jrpls_new_2, jrpls_legacy_2, rnew,
        "JRPLS Prediction (2-component model)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 4: TPLS with 2 components (stable, should match exactly)
    # ========================================================================
    results, tpls_new_2, tpls_legacy_2 = run_tpls_comparison(
        Xi, Ri, process, quality, 2,
        "TPLS (2 components)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 5: TPLS with 4 components (may show numerical drift)
    # ========================================================================
    results, tpls_new_4, tpls_legacy_4 = run_tpls_comparison(
        Xi, Ri, process, quality, 4,
        "TPLS (4 components)"
    )
    all_results.append(results)
    
    # ========================================================================
    # Test 6: TPLS Prediction (using 2-component model for fair comparison)
    # ========================================================================
    # Get process conditions for L001
    znew = process[process['LotID'] == 'L001']
    znew = znew.values.reshape(-1)[1:].astype(float)
    
    results = run_tpls_pred_comparison(
        tpls_new_2, tpls_legacy_2, rnew, znew,
        "TPLS Prediction (2-component model)"
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
        print("\n🎉 SUCCESS! All core JRPLS/TPLS calculations match between implementations.")
        print("   The new refactored code produces equivalent results to the legacy version.")
        return 0
    else:
        print("\n⚠️  Some core tests failed. Please review the differences above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

