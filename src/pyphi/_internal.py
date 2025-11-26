"""
Private helper functions for PyPhi.

This module contains internal implementation details that are not part
of the public API. These functions may change without notice between
minor versions.

Functions prefixed with underscore (_) are strictly internal.
Functions without underscore are semi-internal (used by other modules
but not intended for end-user consumption).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .utils import f95, f99


# =============================================================================
# Statistical Helpers
# =============================================================================


def scores_conf_int_calc(
    st: np.ndarray, N: int, n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate bivariate score confidence interval ellipse points.

    Computes the x and y coordinates for 95% and 99% confidence ellipses
    for bivariate score plots (T1 vs T2).

    Parameters
    ----------
    st : np.ndarray
        2x2 covariance matrix of scores (from first two components).
    N : int
        Number of observations in the training set.
    n_points : int, default=100
        Number of points to generate for each ellipse.

    Returns
    -------
    tuple of np.ndarray
        - xd95: x-coordinates for 95% confidence ellipse
        - xd99: x-coordinates for 99% confidence ellipse
        - yd95p: positive y-coordinates for 95% ellipse
        - yd95n: negative y-coordinates for 95% ellipse
        - yd99p: positive y-coordinates for 99% ellipse
        - yd99n: negative y-coordinates for 99% ellipse
    """
    # Calculate F-distribution critical values with correction factor
    cte2 = ((N - 1) * (N + 1) * 2) / (N * (N - 2))
    f95_ = cte2 * f95(2, N - 2)
    f99_ = cte2 * f99(2, N - 2)

    # X-axis range for ellipse
    xd95 = np.sqrt(f95_ * st[0, 0])
    xd99 = np.sqrt(f99_ * st[0, 0])
    xd95 = np.linspace(-xd95, xd95, num=n_points)
    xd99 = np.linspace(-xd99, xd99, num=n_points)

    # Invert covariance matrix for ellipse calculation
    st_inv = np.linalg.inv(st)
    s11 = st_inv[0, 0]
    s22 = st_inv[1, 1]
    s12 = st_inv[0, 1]
    s21 = st_inv[1, 0]

    # Calculate 95% ellipse y-coordinates using quadratic formula
    a = np.tile(s22, n_points)
    b = xd95 * np.tile(s12, n_points) + xd95 * np.tile(s21, n_points)
    c = (xd95**2) * np.tile(s11, n_points) - f95_
    safe_chk = b**2 - 4 * a * c
    safe_chk[safe_chk < 0] = 0
    yd95p = (-b + np.sqrt(safe_chk)) / (2 * a)
    yd95n = (-b - np.sqrt(safe_chk)) / (2 * a)

    # Calculate 99% ellipse y-coordinates
    a = np.tile(s22, n_points)
    b = xd99 * np.tile(s12, n_points) + xd99 * np.tile(s21, n_points)
    c = (xd99**2) * np.tile(s11, n_points) - f99_
    safe_chk = b**2 - 4 * a * c
    safe_chk[safe_chk < 0] = 0
    yd99p = (-b + np.sqrt(safe_chk)) / (2 * a)
    yd99n = (-b - np.sqrt(safe_chk)) / (2 * a)

    return xd95, xd99, yd95p, yd95n, yd99p, yd99n


def _Ab_btbinv(A: np.ndarray, b: np.ndarray, A_not_nan_map: np.ndarray) -> np.ndarray:
    """Project A onto b with missing data handling: c = Ab / (b'b).

    Computes the projection coefficient for each row of A onto vector b,
    accounting for missing data indicated by A_not_nan_map.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (i, j) to project.
    b : np.ndarray
        Vector of shape (j, 1) to project onto.
    A_not_nan_map : np.ndarray
        Binary matrix of shape (i, j) where 1 indicates non-missing data.

    Returns
    -------
    np.ndarray
        Column vector of shape (i, 1) with projection coefficients.
    """
    b_mat = np.tile(b.T, (A.shape[0], 1))
    c = (np.sum(A * b_mat, axis=1)) / (np.sum((b_mat * A_not_nan_map) ** 2, axis=1))
    return c.reshape(-1, 1)


# =============================================================================
# Pyomo Conversion Helpers
# =============================================================================


def np2D2pyomo(
    arr: np.ndarray, *, varids: list | bool = False
) -> dict[tuple[Any, int], float]:
    """Convert a 2D NumPy array to a dictionary for Pyomo.

    Pyomo requires data in dictionary format with tuple keys for
    indexed parameters. This function converts a NumPy matrix to
    that format.

    Parameters
    ----------
    arr : np.ndarray
        2D array to convert.
    varids : list or False, default=False
        If provided, use these as row indices instead of 1-based integers.

    Returns
    -------
    dict[tuple, float]
        Dictionary with (row_id, col_id) tuple keys and float values.
        Column indices are always 1-based integers.

    Examples
    --------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> np2D2pyomo(arr)
    {(1, 1): 1, (1, 2): 2, (2, 1): 3, (2, 2): 4}
    """
    if not varids:
        output = {
            (i + 1, j + 1): arr[i][j]
            for i in range(arr.shape[0])
            for j in range(arr.shape[1])
        }
    else:
        output = {
            (varids[i], j + 1): arr[i][j]
            for i in range(arr.shape[0])
            for j in range(arr.shape[1])
        }
    return output


def np1D2pyomo(
    arr: np.ndarray, *, indexes: list | bool = False
) -> dict[Any, float]:
    """Convert a 1D NumPy array to a dictionary for Pyomo.

    Pyomo requires data in dictionary format for indexed parameters.
    This function converts a NumPy vector to that format.

    Parameters
    ----------
    arr : np.ndarray
        1D array (or 2D with shape (1, n)) to convert.
    indexes : list or False, default=False
        If provided, use these as indices instead of 1-based integers.

    Returns
    -------
    dict[Any, float]
        Dictionary with index keys and float values.

    Examples
    --------
    >>> arr = np.array([1, 2, 3])
    >>> np1D2pyomo(arr)
    {1: 1, 2: 2, 3: 3}
    """
    if arr.ndim == 2:
        arr = arr[0]
    if isinstance(indexes, bool):
        output = {j + 1: arr[j] for j in range(len(arr))}
    elif isinstance(indexes, list):
        output = {indexes[j]: arr[j] for j in range(len(arr))}
    return output

