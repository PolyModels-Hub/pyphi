"""Utility helpers for PyPhi.

This module provides core utility functions used throughout PyPhi.
Functions have been simplified to use standard library equivalents
from NumPy and SciPy where possible.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import chi2, f as f_dist, t as t_dist


def mean(X: np.ndarray) -> np.ndarray:
    """Column-wise mean that ignores NaN values.

    Parameters
    ----------
    X : np.ndarray
        Input array (2D).

    Returns
    -------
    np.ndarray
        Row vector of column means with shape (1, n_cols).
    """
    return np.nanmean(X, axis=0, keepdims=True)


def std(X: np.ndarray) -> np.ndarray:
    """Column-wise sample standard deviation ignoring NaN values.

    Uses Bessel's correction (ddof=1) for sample standard deviation.

    Parameters
    ----------
    X : np.ndarray
        Input array (2D).

    Returns
    -------
    np.ndarray
        Row vector of column standard deviations with shape (1, n_cols).
    """
    return np.nanstd(X, axis=0, keepdims=True, ddof=1)


def meancenterscale(
    X: np.ndarray, *, mcs: bool | str = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-center and/or scale a data matrix.

    Parameters
    ----------
    X : np.ndarray
        Input data matrix (2D).
    mcs : bool or str, default=True
        Scaling mode:
        - True: mean-center and autoscale (standardize)
        - False: no transformation
        - "center": mean-center only
        - "autoscale": scale by std only (no centering)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        - X_proc: Transformed data matrix
        - x_mean: Mean values used (or np.nan if not centered)
        - x_std: Std values used (or np.nan if not scaled)
    """
    if isinstance(mcs, bool):
        if mcs:
            x_mean = np.nanmean(X, axis=0, keepdims=True)
            x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
            X_proc = (X - x_mean) / x_std
        else:
            X_proc = X
            x_mean = np.nan
            x_std = np.nan
    elif mcs == "center":
        x_mean = np.nanmean(X, axis=0, keepdims=True)
        x_std = np.ones((1, X.shape[1]))
        X_proc = X - x_mean
    elif mcs == "autoscale":
        x_mean = np.zeros((1, X.shape[1]))
        x_std = np.nanstd(X, axis=0, keepdims=True, ddof=1)
        X_proc = X / x_std
    else:
        X_proc = X
        x_mean = np.nan
        x_std = np.nan
    return X_proc, x_mean, x_std


def f95(dfn: float, dfd: float) -> float:
    """F-distribution 95th percentile (critical value for alpha=0.05).

    Parameters
    ----------
    dfn : float
        Numerator degrees of freedom.
    dfd : float
        Denominator degrees of freedom.

    Returns
    -------
    float
        Critical value at 95th percentile.
    """
    return f_dist.ppf(0.95, dfn, dfd)


def f99(dfn: float, dfd: float) -> float:
    """F-distribution 99th percentile (critical value for alpha=0.01).

    Parameters
    ----------
    dfn : float
        Numerator degrees of freedom.
    dfd : float
        Denominator degrees of freedom.

    Returns
    -------
    float
        Critical value at 99th percentile.
    """
    return f_dist.ppf(0.99, dfn, dfd)


# =============================================================================
# NaN Utilities
# =============================================================================


def n2z(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert NaN values to zeros for computation.

    This function replaces NaN values with zeros and returns a map
    of where the NaN values were located, allowing restoration later
    with `z2n()`.

    Parameters
    ----------
    X : np.ndarray
        Input array that may contain NaN values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - X: Modified array with NaN replaced by zeros (modified in place)
        - X_nan_map: Binary array where 1 indicates original NaN positions
    """
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map = X_nan_map.astype(int)
        X[X_nan_map == 1] = 0
    else:
        X_nan_map = X_nan_map.astype(int)
    return X, X_nan_map


def z2n(X: np.ndarray, X_nan_map: np.ndarray) -> np.ndarray:
    """Convert zeros back to NaN using a previously stored NaN map.

    This function restores NaN values to positions indicated by
    X_nan_map, reversing the operation performed by `n2z()`.

    Parameters
    ----------
    X : np.ndarray
        Array with zeros in positions that should be NaN.
    X_nan_map : np.ndarray
        Binary array where 1 indicates positions to restore as NaN.

    Returns
    -------
    np.ndarray
        Array with NaN values restored (modified in place).
    """
    X[X_nan_map == 1] = np.nan
    return X


# =============================================================================
# Statistical Confidence Intervals
# =============================================================================


def spe_ci(spe: np.ndarray) -> tuple[float, float]:
    """Calculate SPE (Squared Prediction Error) confidence intervals.

    Uses the chi-squared distribution approximation for SPE limits
    based on Box's method.

    Parameters
    ----------
    spe : np.ndarray
        Array of SPE values from observations.

    Returns
    -------
    tuple[float, float]
        - lim95: 95% confidence limit
        - lim99: 99% confidence limit
    """
    spe_mean = np.mean(spe)
    if spe_mean > 1e-16:
        spe_var = np.var(spe, ddof=1)
        g = spe_var / (2 * spe_mean)
        h = (2 * spe_mean**2) / spe_var
        lim95 = g * chi2.ppf(0.95, h)
        lim99 = g * chi2.ppf(0.99, h)
    else:
        lim95 = 0.0
        lim99 = 0.0
    return lim95, lim99


def single_score_conf_int(t: np.ndarray) -> tuple[float, float]:
    """Calculate confidence intervals for a single score vector.

    Uses the t-distribution for two-tailed confidence intervals.

    Parameters
    ----------
    t : np.ndarray
        Array of score values from observations.

    Returns
    -------
    tuple[float, float]
        - lim95: 95% confidence limit (two-tailed)
        - lim99: 99% confidence limit (two-tailed)
    """
    n = t.shape[0]
    st = np.var(t, ddof=1)
    # Two-tailed: 0.975 for 95% CI, 0.995 for 99% CI
    lim95 = t_dist.ppf(0.975, n - 1) * np.sqrt(st)
    lim99 = t_dist.ppf(0.995, n - 1) * np.sqrt(st)
    return lim95, lim99


# =============================================================================
# Array Utilities
# =============================================================================


def find(a: np.ndarray, func: Callable) -> list[int]:
    """Find indices where a function/condition is True.

    Parameters
    ----------
    a : np.ndarray
        Input array to search.
    func : Callable
        Function that takes a value and returns True/False.

    Returns
    -------
    list[int]
        List of indices where func(value) is True.

    Examples
    --------
    >>> find(np.array([1, 5, 3, 8, 2]), lambda x: x > 3)
    [1, 3]
    """
    return [i for (i, val) in enumerate(a) if func(val)]


def unique(df: pd.DataFrame, colid: str) -> list:
    """Return unique values in a DataFrame column in order of occurrence.

    Unlike `np.unique()`, this preserves the order in which values
    first appear in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    colid : str
        Column identifier to extract unique values from.

    Returns
    -------
    list
        List of unique values in order of first occurrence.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 2, 1, 3, 2]})
    >>> unique(df, 'A')
    [1, 2, 3]
    """
    return pd.unique(df[colid]).tolist()


# =============================================================================
# Data Cleaning Utilities
# =============================================================================


def clean_htmls() -> None:
    """Remove all HTML files from the current directory.

    Deletes any file with 'html' in its name from the current working
    directory. Useful for cleaning up temporary HTML reports.
    """
    files_here = os.listdir(".")
    for f in files_here:
        if "html" in f:
            os.remove(f)


def clean_empty_rows(
    X: np.ndarray | pd.DataFrame, *, shush: bool = False
) -> tuple[np.ndarray | pd.DataFrame, list[str]]:
    """Remove rows containing all missing data from a matrix.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data matrix to clean. If DataFrame, first column is assumed
        to be observation IDs.
    shush : bool, default=False
        If True, suppress console output about removed rows.

    Returns
    -------
    tuple[np.ndarray | pd.DataFrame, list[str]]
        - X_cleaned: Matrix with empty rows removed
        - rows_removed: List of removed row identifiers
    """
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        ObsID_ = [f"Obs #{n}" for n in range(1, X.shape[0] + 1)]
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str).tolist()

    # Find rows with all data missing
    X_nan_map = np.isnan(X_)
    Xmiss = X_nan_map.astype(int)
    Xmiss = np.sum(Xmiss, axis=1)
    indx = find(Xmiss, lambda x: x == X_.shape[1])

    rows_rem = []
    if len(indx) > 0:
        for i in indx:
            if not shush:
                print(f"Removing row {ObsID_[i]} due to 100% missing data")
            rows_rem.append(ObsID_[i])
        if isinstance(X, pd.DataFrame):
            X_ = X.drop(X.index.values[indx].tolist())
        else:
            X_ = np.delete(X_, indx, 0)
        return X_, rows_rem
    else:
        return X, rows_rem


def clean_low_variances(
    X: np.ndarray | pd.DataFrame, *, shush: bool = False, min_var: float = 1e-10
) -> tuple[np.ndarray | pd.DataFrame, list[str]]:
    """Remove columns with negligible variance from a matrix.

    Removes columns that have:
    1. Too much missing data (>= n_rows - 3 missing values)
    2. Variance below the minimum threshold

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data matrix to clean. If DataFrame, first column is assumed
        to be observation IDs.
    shush : bool, default=False
        If True, suppress console output about removed columns.
    min_var : float, default=1e-10
        Minimum variance threshold. Columns with variance below this
        are removed.

    Returns
    -------
    tuple[np.ndarray | pd.DataFrame, list[str]]
        - X_cleaned: Matrix with low-variance columns removed
        - cols_removed: List of removed column identifiers
    """
    cols_removed = []

    if isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        varidX = X.columns.values[1:].tolist()
    else:
        X_ = X.copy()
        varidX = [f"Var #{n}" for n in range(1, X.shape[1] + 1)]

    # Find columns with too much missing data (must have at least 3 samples)
    X_nan_map = np.isnan(X_)
    Xmiss = X_nan_map.astype(int)
    Xmiss = np.sum(Xmiss, axis=0)

    indx = find(Xmiss, lambda x: x >= (X_.shape[0] - 3))

    if len(indx) > 0:
        for i in indx:
            if not shush:
                print(f"Removing variable {varidX[i]} due to 100% missing data")
        if isinstance(X, pd.DataFrame):
            for i in indx:
                cols_removed.append(varidX[i])
            indx_arr = np.array(indx) + 1
            X_pd = X.drop(X.columns[indx_arr], axis=1)
            X_ = np.array(X_pd.values[:, 1:]).astype(float)
        else:
            for i in indx:
                cols_removed.append(varidX[i])
            X_ = np.delete(X_, indx, 1)
            X_pd = X_  # For numpy, track the cleaned array
    else:
        X_pd = X.copy() if isinstance(X, pd.DataFrame) else X_

    # Get new column names after first cleaning pass
    if isinstance(X, pd.DataFrame):
        new_cols = X_pd.columns[1:].tolist()
    else:
        # For numpy arrays, update varidX by removing dropped columns
        new_cols = [v for i, v in enumerate(varidX) if i not in indx]

    # Find columns with low variance
    std_x = std(X_)
    std_x = std_x.flatten()

    indx2 = find(std_x, lambda x: x < min_var)

    if len(indx2) > 0:
        for i in indx2:
            if not shush:
                print(f"Removing variable {new_cols[i]} due to low variance")
        if isinstance(X, pd.DataFrame):
            for i in indx2:
                cols_removed.append(new_cols[i])
            indx2_arr = np.array(indx2) + 1
            X_ = X_pd.drop(X_pd.columns[indx2_arr], axis=1)
        else:
            for i in indx2:
                cols_removed.append(new_cols[i])
            X_ = np.delete(X_, indx2, 1)
        return X_, cols_removed
    else:
        return X_pd, cols_removed


# =============================================================================
# DataFrame Row/Column Reconciliation
# =============================================================================


def isin_ordered_col0(df: pd.DataFrame, alist: list) -> pd.DataFrame:
    """Filter and reorder DataFrame rows to match a list, using first column.

    Filters the DataFrame to include only rows where the first column
    value is in the provided list, then reorders rows to match the
    list order.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame where first column contains identifiers.
    alist : list
        List of values to filter by and order by.

    Returns
    -------
    pd.DataFrame
        Filtered and reordered DataFrame.
    """
    df_ = df.copy()
    df_ = df_[df_[df_.columns[0]].isin(alist)]
    df_ = df_.set_index(df_.columns[0])
    df_ = df_.reindex(alist)
    df_ = df_.reset_index()
    return df_


def reconcile_rows(df_list: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Reconcile observations across multiple DataFrames.

    Returns a list of DataFrames that all have exactly the same
    observation names (from first column) in the same order.
    Only observations present in ALL DataFrames are kept.

    Useful when analyzing data for TPLS, LPLS, and JRPLS models.

    Parameters
    ----------
    df_list : list[pd.DataFrame]
        List of DataFrames to reconcile. Each DataFrame should have
        observation IDs in the first column.

    Returns
    -------
    list[pd.DataFrame]
        List of DataFrames with reconciled rows.
    """
    # Collect all row identifiers
    all_rows = []
    for df in df_list:
        all_rows.extend(df[df.columns[0]].values.tolist())
    all_rows = np.unique(all_rows)

    # Keep only rows present in ALL DataFrames
    for df in df_list:
        rows = df[df.columns[0]].values.tolist()
        rows_ = []
        for r in all_rows:
            if r in rows:
                rows_.append(r)
        all_rows = rows_.copy()

    # Reorder each DataFrame to match the common rows
    new_df_list = []
    for df in df_list:
        df_ordered = isin_ordered_col0(df, all_rows)
        new_df_list.append(df_ordered)

    return new_df_list


def reconcile_rows_to_columns(
    df_list_r: list[pd.DataFrame], df_list_c: list[pd.DataFrame]
) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Reconcile rows of one DataFrame list with columns of another.

    For each pair of DataFrames, aligns the row identifiers in df_list_r
    with the column identifiers in df_list_c. Only keeps identifiers
    present in both rows AND columns.

    Useful to align X-R datasets for TPLS, LPLS, and JRPLS models.

    Parameters
    ----------
    df_list_r : list[pd.DataFrame]
        List of DataFrames with row identifiers in first column.
    df_list_c : list[pd.DataFrame]
        List of DataFrames with column identifiers to align with.

    Returns
    -------
    tuple[list[pd.DataFrame], list[pd.DataFrame]]
        - df_list_r_out: Reconciled row DataFrames
        - df_list_c_out: Reconciled column DataFrames
    """
    df_list_r_o = []
    df_list_c_o = []

    for dfr, dfc in zip(df_list_r, df_list_c):
        # Get all identifiers from both sources
        all_ids = dfc.columns[1:].tolist()
        all_ids.extend(dfr[dfr.columns[0]].values.tolist())
        all_ids = np.unique(all_ids)

        # Keep only IDs present in rows
        rows = dfr[dfr.columns[0]].values.tolist()
        cols = dfc.columns[1:].tolist()

        all_ids_ = []
        for i in all_ids:
            if i in rows:
                all_ids_.append(i)

        # Keep only IDs present in columns
        all_ids_final = []
        for i in all_ids_:
            if i in cols:
                all_ids_final.append(i)

        # Reorder row DataFrame
        dfr_ = isin_ordered_col0(dfr, all_ids_final)

        # Reorder column DataFrame
        dfc_ = dfc[all_ids_final].copy()
        dfc_.insert(0, dfc.columns[0], dfc[dfc.columns[0]].values.tolist())

        df_list_r_o.append(dfr_)
        df_list_c_o.append(dfc_)

    return df_list_r_o, df_list_c_o
