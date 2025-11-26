"""
Partial Least Squares (PLS) module for PyPhi.

This module implements PLS regression with support for:
- SVD algorithm for complete data
- NIPALS algorithm for data with missing values
- NLP optimization for missing data (via Pyomo)
- Cross-validation for model selection
- CCA (Canonical Correlation Analysis) for OPLS-like behavior

References:
    - Wold et al. (2001) PLS-regression: a basic tool of chemometrics
    - Lopez-Negrete et al. J. Chemometrics 2010; 24: 301-311 (NLP approach)
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

from .utils import (
    meancenterscale,
    n2z,
    z2n,
    std,
    f95,
    f99,
    spe_ci,
)
from ._internal import np2D2pyomo
from .pca import hott2, pca_pred

# Check for Pyomo availability
try:
    from pyomo.environ import (
        ConcreteModel,
        Set,
        Var,
        Param,
        Constraint,
        Objective,
        Reals,
        SolverFactory,
        SolverManagerFactory,
        value,
    )
    _PYOMO_AVAILABLE = True
except ImportError:
    _PYOMO_AVAILABLE = False


# =============================================================================
# CCA Functions
# =============================================================================


def cca(
    X: np.ndarray,
    Y: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Perform Canonical Correlation Analysis (CCA) on two datasets.

    Parameters
    ----------
    X : np.ndarray
        An (n x p) matrix where n is the number of samples.
    Y : np.ndarray
        An (n x q) matrix where n is the number of samples.
    tol : float, default=1e-6
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    tuple
        (correlation, w_x, w_y) - canonical correlation and direction vectors.
    """
    # Center the matrices
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Compute covariance matrices
    Sigma_XX = np.dot(X.T, X)
    Sigma_YY = np.dot(Y.T, Y)
    Sigma_XY = np.dot(X.T, Y)

    # Initialize random vectors
    w_x = np.random.rand(X.shape[1])
    w_y = np.random.rand(Y.shape[1])

    # Normalize initial vectors
    w_x /= np.linalg.norm(w_x)
    w_y /= np.linalg.norm(w_y)

    # Iteratively update
    for iteration in range(max_iter):
        w_x_old = w_x.copy()
        w_y_old = w_y.copy()

        # Update w_x and normalize
        w_x = np.linalg.solve(Sigma_XX, Sigma_XY @ w_y)
        w_x /= np.linalg.norm(w_x)

        # Update w_y and normalize
        w_y = np.linalg.solve(Sigma_YY, Sigma_XY.T @ w_x)
        w_y /= np.linalg.norm(w_y)

        # Calculate correlation
        correlation = w_x.T @ Sigma_XY @ w_y

        # Check convergence
        if np.linalg.norm(w_x - w_x_old) < tol and np.linalg.norm(w_y - w_y_old) < tol:
            break

    return correlation, w_x, w_y


def cca_multi(
    X: np.ndarray,
    Y: np.ndarray,
    num_components: int = 1,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> dict:
    """Perform CCA to compute multiple canonical variates.

    Parameters
    ----------
    X : np.ndarray
        An (n x p) matrix.
    Y : np.ndarray
        An (n x q) matrix.
    num_components : int, default=1
        Number of canonical variates to compute.
    tol : float, default=1e-6
        Tolerance for convergence.
    max_iter : int, default=1000
        Maximum number of iterations.

    Returns
    -------
    dict
        Contains correlations and canonical direction vectors for X and Y.
    """
    # Center the matrices
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    correlations = []
    W_X = []
    W_Y = []

    for component in range(num_components):
        Sigma_XX = np.dot(X.T, X)
        Sigma_YY = np.dot(Y.T, Y)
        Sigma_XY = np.dot(X.T, Y)

        w_x = np.random.rand(X.shape[1])
        w_y = np.random.rand(Y.shape[1])
        w_x /= np.linalg.norm(w_x)
        w_y /= np.linalg.norm(w_y)

        for iteration in range(max_iter):
            w_x_old = w_x.copy()
            w_y_old = w_y.copy()

            w_x = np.linalg.solve(Sigma_XX, Sigma_XY @ w_y)
            w_x /= np.linalg.norm(w_x)

            w_y = np.linalg.solve(Sigma_YY, Sigma_XY.T @ w_x)
            w_y /= np.linalg.norm(w_y)

            correlation = w_x.T @ Sigma_XY @ w_y

            if np.linalg.norm(w_x - w_x_old) < tol and np.linalg.norm(w_y - w_y_old) < tol:
                break

        correlations.append(correlation)
        W_X.append(w_x)
        W_Y.append(w_y)

        # Deflate X and Y
        X -= np.dot(X @ w_x[:, np.newaxis], w_x[np.newaxis, :])
        Y -= np.dot(Y @ w_y[:, np.newaxis], w_y[np.newaxis, :])

    return {
        "correlations": np.array(correlations),
        "W_X": np.array(W_X).T,
        "W_Y": np.array(W_Y).T,
    }


def _pls_cca(
    pls_obj: dict,
    Xmcs: np.ndarray,
    Ymcs: np.ndarray,
    not_Xmiss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate covariant scores and loadings for PLS-CCA.

    Internal function used when cca=True in pls().
    """
    Tcv = []
    Pcv = []
    Wcv = []
    Betacv = []
    firstone = True

    for i in range(Ymcs.shape[1]):
        y_ = Ymcs[:, i].reshape(-1, 1)
        corr, wt, wy = cca(pls_obj["T"], y_)
        t_cv = pls_obj["T"] @ wt
        t_cv = t_cv.reshape(-1, 1)
        beta = np.linalg.lstsq(t_cv, y_, rcond=None)[0]
        w_cv = pls_obj["Ws"] @ wt
        w_cv = w_cv.reshape(-1, 1)
        tcvmat = np.tile(t_cv, (1, Xmcs.shape[1]))
        p_cv = np.sum(Xmcs * tcvmat, axis=0) / np.sum((tcvmat * not_Xmiss) ** 2, axis=0)
        p_cv = p_cv.reshape(-1, 1)

        if firstone:
            Tcv = t_cv
            Pcv = p_cv
            Wcv = w_cv
            Betacv = beta[0][0]
            firstone = False
        else:
            Tcv = np.hstack((Tcv, t_cv))
            Pcv = np.hstack((Pcv, p_cv))
            Wcv = np.hstack((Wcv, w_cv))
            Betacv = np.vstack((Betacv, beta[0][0]))

    return Tcv, Pcv, Wcv, Betacv


# =============================================================================
# SPE Function
# =============================================================================


def spe(
    mvmobj: dict,
    Xnew: np.ndarray | pd.DataFrame,
    *,
    Ynew: np.ndarray | pd.DataFrame | bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calculate Squared Prediction Error (SPE).

    Parameters
    ----------
    mvmobj : dict
        PCA or PLS model object.
    Xnew : np.ndarray or pd.DataFrame
        New X data.
    Ynew : np.ndarray, pd.DataFrame, or False, default=False
        New Y data (only for PLS models).

    Returns
    -------
    np.ndarray or tuple
        SPE for X, and optionally SPE for Y (for PLS models with Ynew).
    """
    if "Q" in mvmobj:
        xpred = pls_pred(Xnew, mvmobj)
    else:
        xpred = pca_pred(Xnew, mvmobj)
    Tnew = xpred["Tnew"]

    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:, 1:]).astype(float)

    if isinstance(Ynew, np.ndarray):
        Y_ = Ynew.copy()
    elif isinstance(Ynew, pd.DataFrame):
        Y_ = np.array(Ynew.values[:, 1:]).astype(float)

    Xnewhat = Tnew @ mvmobj["P"].T
    Xres = X_ - np.tile(mvmobj["mx"], (X_.shape[0], 1))
    Xres = Xres / np.tile(mvmobj["sx"], (X_.shape[0], 1))
    Xres = Xres - Xnewhat
    spex_ = np.sum(Xres**2, axis=1, keepdims=True)

    if not isinstance(Ynew, bool) and ("Q" in mvmobj):
        Ynewhat = Tnew @ mvmobj["Q"].T
        Yres = Y_ - np.tile(mvmobj["my"], (Y_.shape[0], 1))
        Yres = Yres / np.tile(mvmobj["sy"], (Y_.shape[0], 1))
        Yres = Yres - Ynewhat
        spey_ = np.sum(Yres**2, axis=1, keepdims=True)
        return spex_, spey_
    else:
        return spex_


# =============================================================================
# Prep for NLP
# =============================================================================


def prep_pls_4_MDbyNLP(plsobj: dict, X: np.ndarray, Y: np.ndarray) -> dict:
    """Prepare PLS object for missing data handling via NLP.

    Parameters
    ----------
    plsobj : dict
        PLS model object from pls_().
    X : np.ndarray
        X data matrix (may contain NaN).
    Y : np.ndarray
        Y data matrix (may contain NaN).

    Returns
    -------
    dict
        Augmented PLS object with Pyomo-formatted parameters.
    """
    plsobj_ = plsobj.copy()
    X_nan_map = np.isnan(X)
    psi = (~X_nan_map).astype(int)
    X, dummy = n2z(X.copy())

    Y_nan_map = np.isnan(Y)
    theta = (~Y_nan_map).astype(int)
    Y, dummy = n2z(Y.copy())

    A = plsobj["T"].shape[1]
    O = plsobj["T"].shape[0]
    N = plsobj["P"].shape[0]
    M = plsobj["Q"].shape[0]

    pyo_A = list(range(1, A + 1))
    pyo_N = list(range(1, N + 1))
    pyo_O = list(range(1, O + 1))
    pyo_M = list(range(1, M + 1))

    pyo_P_init = np2D2pyomo(plsobj["P"])
    pyo_Ws_init = np2D2pyomo(plsobj["Ws"])
    pyo_T_init = np2D2pyomo(plsobj["T"])
    pyo_Q_init = np2D2pyomo(plsobj["Q"])
    pyo_X = np2D2pyomo(X)
    pyo_Y = np2D2pyomo(Y)
    pyo_psi = np2D2pyomo(psi)
    pyo_theta = np2D2pyomo(theta)

    plsobj_["pyo_A"] = pyo_A
    plsobj_["pyo_N"] = pyo_N
    plsobj_["pyo_O"] = pyo_O
    plsobj_["pyo_M"] = pyo_M
    plsobj_["pyo_P_init"] = pyo_P_init
    plsobj_["pyo_Ws_init"] = pyo_Ws_init
    plsobj_["pyo_T_init"] = pyo_T_init
    plsobj_["pyo_Q_init"] = pyo_Q_init
    plsobj_["pyo_X"] = pyo_X
    plsobj_["pyo_psi"] = pyo_psi
    plsobj_["pyo_Y"] = pyo_Y
    plsobj_["pyo_theta"] = pyo_theta

    return plsobj_


# =============================================================================
# PLS Prediction
# =============================================================================


def pls_pred(
    Xnew: np.ndarray | pd.DataFrame | dict,
    plsobj: dict,
) -> dict:
    """Evaluate new data using an already built PLS model.

    Parameters
    ----------
    Xnew : np.ndarray, pd.DataFrame, or dict
        New X data to project. If dict, expects multi-block format.
    plsobj : dict
        PLS model object created by pls().

    Returns
    -------
    dict
        Prediction results containing:
        - Yhat: Predicted Y values
        - Xhat: Reconstructed X values
        - Tnew: Projected scores
        - speX: SPE for X
        - T2: Hotelling's T²
        - Tcv: (if CCA was used) Covariant scores
    """
    algorithm = "p2mp"
    force_deflation = False

    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
        if X_.ndim == 1:
            X_ = np.reshape(X_, (1, -1))
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:, 1:]).astype(float)
    elif isinstance(Xnew, dict):
        # Multi-block data
        data_ = []
        names_ = []
        for k in Xnew.keys():
            data_.append(Xnew[k])
            names_.append(k)
        XMB = {"data": data_, "blknames": names_}
        c = 0
        for i, x in enumerate(XMB["data"]):
            x_ = x.values[:, 1:].astype(float)
            if c == 0:
                X_ = x_.copy()
            else:
                X_ = np.hstack((X_, x_))
            c += 1

    X_nan_map = np.isnan(X_)

    if not X_nan_map.any() and not force_deflation:
        # Complete data - direct projection
        tnew = (
            (X_ - np.tile(plsobj["mx"], (X_.shape[0], 1)))
            / np.tile(plsobj["sx"], (X_.shape[0], 1))
        ) @ plsobj["Ws"]
        yhat = (tnew @ plsobj["Q"].T) * np.tile(
            plsobj["sy"], (X_.shape[0], 1)
        ) + np.tile(plsobj["my"], (X_.shape[0], 1))
        xhat = (tnew @ plsobj["P"].T) * np.tile(
            plsobj["sx"], (X_.shape[0], 1)
        ) + np.tile(plsobj["mx"], (X_.shape[0], 1))
        var_t = (plsobj["T"].T @ plsobj["T"]) / plsobj["T"].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        speX = (
            (X_ - np.tile(plsobj["mx"], (X_.shape[0], 1)))
            / np.tile(plsobj["sx"], (X_.shape[0], 1))
        ) - (tnew @ plsobj["P"].T)
        speX = np.sum(speX**2, axis=1, keepdims=True)
        ypred = {"Yhat": yhat, "Xhat": xhat, "Tnew": tnew, "speX": speX, "T2": htt2}

    elif algorithm == "p2mp":
        # Projection to Model Plane for missing data
        not_Xmiss = (~X_nan_map).astype(int)
        Xmcs = (X_ - np.tile(plsobj["mx"], (X_.shape[0], 1))) / np.tile(
            plsobj["sx"], (X_.shape[0], 1)
        )
        Xmcs, dummy = n2z(Xmcs.copy())

        for i in range(Xmcs.shape[0]):
            row_missing_map = not_Xmiss[[i], :]
            tempW = plsobj["W"] * np.tile(row_missing_map.T, (1, plsobj["W"].shape[1]))

            for a in range(plsobj["W"].shape[1]):
                WTW = tempW[:, [a]].T @ tempW[:, [a]]
                tnew_aux, resid, rank, s = np.linalg.lstsq(
                    WTW, (tempW[:, [a]].T @ Xmcs[[i], :].T), rcond=None
                )
                Xmcs[[i], :] = (
                    Xmcs[[i], :] - tnew_aux @ plsobj["P"][:, [a]].T
                ) * row_missing_map
                if a == 0:
                    tnew_ = tnew_aux
                else:
                    tnew_ = np.vstack((tnew_, tnew_aux))

            if i == 0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew, tnew_.T))

        yhat = (tnew @ plsobj["Q"].T) * np.tile(
            plsobj["sy"], (X_.shape[0], 1)
        ) + np.tile(plsobj["my"], (X_.shape[0], 1))
        xhat = (tnew @ plsobj["P"].T) * np.tile(
            plsobj["sx"], (X_.shape[0], 1)
        ) + np.tile(plsobj["mx"], (X_.shape[0], 1))
        var_t = (plsobj["T"].T @ plsobj["T"]) / plsobj["T"].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        X_, dummy = n2z(X_.copy())
        speX = (
            (X_ - np.tile(plsobj["mx"], (X_.shape[0], 1)))
            / np.tile(plsobj["sx"], (X_.shape[0], 1))
        ) - (tnew @ plsobj["P"].T)
        speX = speX * not_Xmiss
        speX = np.sum(speX**2, axis=1, keepdims=True)
        ypred = {"Yhat": yhat, "Xhat": xhat, "Tnew": tnew, "speX": speX, "T2": htt2}

        if "Wcv" in plsobj:
            Tcv = (
                (X_ - np.tile(plsobj["mx"], (X_.shape[0], 1)))
                / np.tile(plsobj["sx"], (X_.shape[0], 1))
            ) @ plsobj["Wcv"]
            ypred["Tcv"] = Tcv

    return ypred


# =============================================================================
# Core PLS Algorithm
# =============================================================================


def pls_(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcsX: bool | str = True,
    mcsY: bool | str = True,
    md_algorithm: str = "nipals",
    force_nipals: bool = True,
    shush: bool = False,
    cca_flag: bool = False,
) -> dict:
    """Core PLS algorithm implementation.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Predictor data matrix.
    Y : np.ndarray or pd.DataFrame
        Response data matrix.
    A : int
        Number of Latent Variables to calculate.
    mcsX : bool or str, default=True
        X preprocessing mode.
    mcsY : bool or str, default=True
        Y preprocessing mode.
    md_algorithm : str, default="nipals"
        Missing data algorithm ("nipals" or "nlp").
    force_nipals : bool, default=True
        Force NIPALS even for complete data.
    shush : bool, default=False
        Suppress output.
    cca_flag : bool, default=False
        Calculate CCA for OPLS-like behavior.

    Returns
    -------
    dict
        PLS model object.
    """
    # Extract X data and identifiers
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        obsidX = X.values[:, 0].astype(str).tolist()
        varidX = X.columns.values[1:].tolist()

    # Extract Y data and identifiers
    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
        obsidY = False
        varidY = False
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)
        obsidY = Y.values[:, 0].astype(str).tolist()
        varidY = Y.columns.values[1:].tolist()

    # Preprocess X
    if isinstance(mcsX, bool):
        if mcsX:
            X_, x_mean, x_std = meancenterscale(X_)
        else:
            x_mean = np.zeros((1, X_.shape[1]))
            x_std = np.ones((1, X_.shape[1]))
    elif mcsX == "center":
        X_, x_mean, x_std = meancenterscale(X_, mcs="center")
    elif mcsX == "autoscale":
        X_, x_mean, x_std = meancenterscale(X_, mcs="autoscale")

    # Preprocess Y
    if isinstance(mcsY, bool):
        if mcsY:
            Y_, y_mean, y_std = meancenterscale(Y_)
        else:
            y_mean = np.zeros((1, Y_.shape[1]))
            y_std = np.ones((1, Y_.shape[1]))
    elif mcsY == "center":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="center")
    elif mcsY == "autoscale":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="autoscale")

    # Generate missing data maps
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map).astype(int)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (~Y_nan_map).astype(int)

    # Choose algorithm
    if not X_nan_map.any() and not Y_nan_map.any() and not force_nipals:
        return _pls_svd(
            X_, Y_, A, x_mean, x_std, y_mean, y_std,
            not_Xmiss, obsidX, varidX, obsidY, varidY, shush, cca_flag
        )
    elif md_algorithm == "nipals":
        return _pls_nipals(
            X_, Y_, A, x_mean, x_std, y_mean, y_std,
            not_Xmiss, not_Ymiss, obsidX, varidX, obsidY, varidY, shush, cca_flag
        )
    elif md_algorithm == "nlp" and _PYOMO_AVAILABLE:
        return _pls_nlp(
            X, Y, X_, Y_, A, mcsX, mcsY, x_mean, x_std, y_mean, y_std,
            X_nan_map, not_Xmiss, not_Ymiss, obsidX, varidX, obsidY, varidY, shush
        )
    elif md_algorithm == "nlp" and not _PYOMO_AVAILABLE:
        print("Pyomo was not found in your system")
        print("visit http://www.pyomo.org/")
        return {"error": "Pyomo not available"}


def _pls_svd(
    X_: np.ndarray,
    Y_: np.ndarray,
    A: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    not_Xmiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    obsidY: list | bool,
    varidY: list | bool,
    shush: bool,
    cca_flag: bool,
) -> dict:
    """PLS using SVD for complete data."""
    if not shush:
        print(f"phi.pls using SVD executed on: {datetime.datetime.now()}")

    if cca_flag:
        Xmcs = X_.copy()
        Ymcs = Y_.copy()

    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    for a in range(A):
        U_, S, Wh = np.linalg.svd((X_.T @ Y_) @ (Y_.T @ X_))
        w = Wh.T[:, [0]]
        t = X_ @ w
        q = Y_.T @ t / (t.T @ t)
        u = Y_ @ q / (q.T @ q)
        p = X_.T @ t / (t.T @ t)

        X_ = X_ - t @ p.T
        Y_ = Y_ - t @ q.T

        if a == 0:
            W = w.reshape(-1, 1)
            T = t.reshape(-1, 1)
            Q = q.reshape(-1, 1)
            U = u.reshape(-1, 1)
            P = p.reshape(-1, 1)

            r2X = 1 - np.sum(X_**2) / TSSX
            r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
            r2Y = 1 - np.sum(Y_**2) / TSSY
            r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
        else:
            W = np.hstack((W, w.reshape(-1, 1)))
            T = np.hstack((T, t.reshape(-1, 1)))
            Q = np.hstack((Q, q.reshape(-1, 1)))
            U = np.hstack((U, u.reshape(-1, 1)))
            P = np.hstack((P, p.reshape(-1, 1)))

            r2X_ = 1 - np.sum(X_**2) / TSSX
            r2Xpv_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
            r2X = np.hstack((r2X, r2X_))
            r2Xpv = np.hstack((r2Xpv, r2Xpv_))

            r2Y_ = 1 - np.sum(Y_**2) / TSSY
            r2Ypv_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
            r2Y = np.hstack((r2Y, r2Y_))
            r2Ypv = np.hstack((r2Ypv, r2Ypv_))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2X[a] = r2X[a] - r2X[a - 1]
        r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]

    Ws = W @ np.linalg.pinv(P.T @ W)
    Ws[:, 0] = W[:, 0]

    eigs = np.var(T, axis=0)
    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)

    if not shush:
        print("-" * 62)
        print("LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)")
        if A > 1:
            for a in range(A):
                print(
                    f"LV #{a + 1}:   {eigs[a]:6.3f}    {r2X[a]:.3f}     "
                    f"{r2xc[a]:.3f}      {r2Y[a]:.3f}     {r2yc[a]:.3f}"
                )
        else:
            print(
                f"LV #1:   {eigs[0]:6.3f}    {r2X:.3f}     "
                f"{r2xc[0]:.3f}      {r2Y:.3f}     {r2yc[0]:.3f}"
            )
        print("-" * 62)

    pls_obj = {
        "T": T, "P": P, "Q": Q, "W": W, "Ws": Ws, "U": U,
        "r2x": r2X, "r2xpv": r2Xpv, "mx": x_mean, "sx": x_std,
        "r2y": r2Y, "r2ypv": r2Ypv, "my": y_mean, "sy": y_std,
    }

    if not isinstance(obsidX, bool):
        pls_obj["obsidX"] = obsidX
        pls_obj["varidX"] = varidX
    if not isinstance(obsidY, bool):
        pls_obj["obsidY"] = obsidY
        pls_obj["varidY"] = varidY

    # Add diagnostics
    _add_pls_diagnostics(pls_obj, T, X_, Y_, A)

    if cca_flag:
        Tcv, Pcv, Wcv, Betacv = _pls_cca(pls_obj, Xmcs, Ymcs, not_Xmiss)
        pls_obj["Tcv"] = Tcv
        pls_obj["Pcv"] = Pcv
        pls_obj["Wcv"] = Wcv
        pls_obj["Betacv"] = Betacv

    return pls_obj


def _pls_nipals(
    X_: np.ndarray,
    Y_: np.ndarray,
    A: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    not_Xmiss: np.ndarray,
    not_Ymiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    obsidY: list | bool,
    varidY: list | bool,
    shush: bool,
    cca_flag: bool,
) -> dict:
    """PLS using NIPALS algorithm (handles missing data)."""
    if not shush:
        print(f"phi.pls using NIPALS executed on: {datetime.datetime.now()}")

    X_, dummy = n2z(X_.copy())
    Y_, dummy = n2z(Y_.copy())
    epsilon = 1e-9
    maxit = 2000

    if cca_flag:
        Xmcs = X_.copy()
        Ymcs = Y_.copy()

    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    for a in range(A):
        # Initial guess: column with largest variance in Y
        ui = Y_[:, [np.argmax(std(Y_))]]
        converged = False
        num_it = 0

        while not converged:
            # Step 1: w = X'u / u'u
            uimat = np.tile(ui, (1, X_.shape[1]))
            wi = np.sum(X_ * uimat, axis=0) / np.sum((uimat * not_Xmiss) ** 2, axis=0)

            # Step 2: Normalize w
            wi = wi / np.linalg.norm(wi)

            # Step 3: t = Xw / w'w
            wimat = np.tile(wi, (X_.shape[0], 1))
            ti = X_ @ wi.T
            wtw = np.sum((wimat * not_Xmiss) ** 2, axis=1)
            ti = ti / wtw
            ti = ti.reshape(-1, 1)
            wi = wi.reshape(-1, 1)

            # Step 4: q = Y't / t't
            timat = np.tile(ti, (1, Y_.shape[1]))
            qi = np.sum(Y_ * timat, axis=0) / np.sum((timat * not_Ymiss) ** 2, axis=0)

            # Step 5: u_new = Yq / q'q
            qimat = np.tile(qi, (Y_.shape[0], 1))
            qi = qi.reshape(-1, 1)
            un = Y_ @ qi
            qtq = np.sum((qimat * not_Ymiss) ** 2, axis=1)
            qtq = qtq.reshape(-1, 1)
            un = un / qtq
            un = un.reshape(-1, 1)

            if abs(np.linalg.norm(ui) - np.linalg.norm(un)) / np.linalg.norm(ui) < epsilon:
                converged = True
            if num_it > maxit:
                converged = True

            if converged:
                if np.var(ti[ti < 0]) > np.var(ti[ti >= 0]):
                    ti = -ti
                    wi = -wi
                    un = -un
                    qi = -qi

                if not shush:
                    print(f"# Iterations for LV #{a + 1}: {num_it}")

                # Calculate P for deflation
                timat = np.tile(ti, (1, X_.shape[1]))
                pi = np.sum(X_ * timat, axis=0) / np.sum((timat * not_Xmiss) ** 2, axis=0)
                pi = pi.reshape(-1, 1)

                # Deflate
                X_ = (X_ - ti @ pi.T) * not_Xmiss
                Y_ = (Y_ - ti @ qi.T) * not_Ymiss

                if a == 0:
                    T = ti
                    P = pi
                    W = wi
                    U = un
                    Q = qi
                    r2X = 1 - np.sum(X_**2) / TSSX
                    r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                    r2Y = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                else:
                    T = np.hstack((T, ti.reshape(-1, 1)))
                    U = np.hstack((U, un.reshape(-1, 1)))
                    P = np.hstack((P, pi))
                    W = np.hstack((W, wi))
                    Q = np.hstack((Q, qi))

                    r2X_ = 1 - np.sum(X_**2) / TSSX
                    r2Xpv_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                    r2X = np.hstack((r2X, r2X_))
                    r2Xpv = np.hstack((r2Xpv, r2Xpv_))

                    r2Y_ = 1 - np.sum(Y_**2) / TSSY
                    r2Ypv_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                    r2Y = np.hstack((r2Y, r2Y_))
                    r2Ypv = np.hstack((r2Ypv, r2Ypv_))
            else:
                num_it += 1
                ui = un

        if a == 0:
            numIT = num_it
        else:
            numIT = np.hstack((numIT, num_it))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2X[a] = r2X[a] - r2X[a - 1]
        r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]

    Ws = W @ np.linalg.pinv(P.T @ W)
    Ws[:, 0] = W[:, 0]

    eigs = np.var(T, axis=0)
    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)

    if not shush:
        print("-" * 62)
        print("LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)")
        if A > 1:
            for a in range(A):
                print(
                    f"LV #{a + 1}:   {eigs[a]:6.3f}    {r2X[a]:.3f}     "
                    f"{r2xc[a]:.3f}      {r2Y[a]:.3f}     {r2yc[a]:.3f}"
                )
        else:
            print(
                f"LV #1:   {eigs[0]:6.3f}    {r2X:.3f}     "
                f"{r2xc[0]:.3f}      {r2Y:.3f}     {r2yc[0]:.3f}"
            )
        print("-" * 62)

    pls_obj = {
        "T": T, "P": P, "Q": Q, "W": W, "Ws": Ws, "U": U,
        "r2x": r2X, "r2xpv": r2Xpv, "mx": x_mean, "sx": x_std,
        "r2y": r2Y, "r2ypv": r2Ypv, "my": y_mean, "sy": y_std,
    }

    if not isinstance(obsidX, bool):
        pls_obj["obsidX"] = obsidX
        pls_obj["varidX"] = varidX
    if not isinstance(obsidY, bool):
        pls_obj["obsidY"] = obsidY
        pls_obj["varidY"] = varidY

    # Add diagnostics
    _add_pls_diagnostics(pls_obj, T, X_, Y_, A)

    if cca_flag:
        Tcv, Pcv, Wcv, Betacv = _pls_cca(pls_obj, Xmcs, Ymcs, not_Xmiss)
        pls_obj["Tcv"] = Tcv
        pls_obj["Pcv"] = Pcv
        pls_obj["Wcv"] = Wcv
        pls_obj["Betacv"] = Betacv

    return pls_obj


def _pls_nlp(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    X_: np.ndarray,
    Y_: np.ndarray,
    A: int,
    mcsX: bool | str,
    mcsY: bool | str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    y_mean: np.ndarray,
    y_std: np.ndarray,
    X_nan_map: np.ndarray,
    not_Xmiss: np.ndarray,
    not_Ymiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    obsidY: list | bool,
    varidY: list | bool,
    shush: bool,
) -> dict:
    """PLS using NLP optimization for missing data."""
    from shutil import which

    if not shush:
        print(f"phi.pls using NLP with Ipopt executed on: {datetime.datetime.now()}")

    X_, dummy = n2z(X_.copy())
    Y_, dummy = n2z(Y_.copy())

    # Get initial solution from NIPALS
    plsobj_ = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, md_algorithm="nipals", shush=True)
    plsobj_ = prep_pls_4_MDbyNLP(plsobj_, X_, Y_)

    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)

    # Set up Pyomo model for T and P
    model = ConcreteModel()
    model.A = Set(initialize=plsobj_["pyo_A"])
    model.N = Set(initialize=plsobj_["pyo_N"])
    model.M = Set(initialize=plsobj_["pyo_M"])
    model.O = Set(initialize=plsobj_["pyo_O"])
    model.P = Var(model.N, model.A, within=Reals, initialize=plsobj_["pyo_P_init"])
    model.T = Var(model.O, model.A, within=Reals, initialize=plsobj_["pyo_T_init"])
    model.psi = Param(model.O, model.N, initialize=plsobj_["pyo_psi"])
    model.X = Param(model.O, model.N, initialize=plsobj_["pyo_X"])
    model.theta = Param(model.O, model.M, initialize=plsobj_["pyo_theta"])
    model.Y = Param(model.O, model.M, initialize=plsobj_["pyo_Y"])
    model.delta = Param(
        model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1 == a2 else 0
    )

    # Constraints
    def _c27bc_con(model, a1, a2):
        return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]
    model.c27bc = Constraint(model.A, model.A, rule=_c27bc_con)

    def _27d_con(model, a1, a2):
        if a2 < a1:
            return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
        else:
            return Constraint.Skip
    model.c27d = Constraint(model.A, model.A, rule=_27d_con)

    def _27e_con(model, i):
        return sum(model.T[o, i] for o in model.O) == 0
    model.c27e = Constraint(model.A, rule=_27e_con)

    def _eq_27a_obj(model):
        return sum(
            sum(
                sum(
                    (model.theta[o, m] * model.Y[o, m])
                    * (model.X[o, n] - model.psi[o, n] * sum(model.T[o, a] * model.P[n, a] for a in model.A))
                    for o in model.O
                ) ** 2
                for n in model.N
            )
            for m in model.M
        )
    model.obj = Objective(rule=_eq_27a_obj)

    # Solve
    ipopt_ok = bool(which("ipopt"))
    gams_ok = bool(which("gams"))

    if ipopt_ok:
        print("Solving NLP using local IPOPT executable")
        solver = SolverFactory("ipopt")
        results = solver.solve(model, tee=True)
    elif gams_ok:
        print("Solving NLP using GAMS/IPOPT interface")
        solver = SolverFactory("gams:ipopt")
        results = solver.solve(model, tee=True)
    else:
        print("Solving NLP using IPOPT on remote NEOS server")
        solver_manager = SolverManagerFactory("neos")
        results = solver_manager.solve(model, opt="ipopt", tee=True)

    # Extract T and P
    T = np.array([[value(model.T[o, a]) for a in model.A] for o in model.O])
    P = np.array([[value(model.P[n, a]) for a in model.A] for n in model.N])

    # Now solve for Ws
    Taux = np2D2pyomo(T)
    modelb = ConcreteModel()
    modelb.A = Set(initialize=plsobj_["pyo_A"])
    modelb.N = Set(initialize=plsobj_["pyo_N"])
    modelb.O = Set(initialize=plsobj_["pyo_O"])
    modelb.Ws = Var(model.N, model.A, within=Reals, initialize=plsobj_["pyo_Ws_init"])
    modelb.T = Param(model.O, model.A, within=Reals, initialize=Taux)
    modelb.psi = Param(model.O, model.N, initialize=plsobj_["pyo_psi"])
    modelb.X = Param(model.O, model.N, initialize=plsobj_["pyo_X"])

    def _eq_obj(model):
        return sum(
            sum(
                (model.T[o, a] - sum(model.psi[o, n] * model.X[o, n] * model.Ws[n, a] for n in model.N)) ** 2
                for a in model.A
            )
            for o in model.O
        )
    modelb.obj = Objective(rule=_eq_obj)

    if ipopt_ok:
        solver = SolverFactory("ipopt")
        results = solver.solve(modelb, tee=True)
    elif gams_ok:
        solver = SolverFactory("gams:ipopt")
        results = solver.solve(modelb, tee=True)
    else:
        solver_manager = SolverManagerFactory("neos")
        results = solver_manager.solve(modelb, opt="ipopt", tee=True)

    Ws = np.array([[value(modelb.Ws[n, a]) for a in modelb.A] for n in modelb.N])

    # Now solve for Q
    Xhat = T @ P.T
    Xaux = X_.copy()
    Xaux[X_nan_map] = Xhat[X_nan_map]
    Xaux = np2D2pyomo(Xaux)
    Taux = np2D2pyomo(T)

    model2 = ConcreteModel()
    model2.A = Set(initialize=plsobj_["pyo_A"])
    model2.N = Set(initialize=plsobj_["pyo_N"])
    model2.M = Set(initialize=plsobj_["pyo_M"])
    model2.O = Set(initialize=plsobj_["pyo_O"])
    model2.T = Param(model.O, model.A, within=Reals, initialize=Taux)
    model2.Q = Var(model.M, model.A, within=Reals, initialize=plsobj_["pyo_Q_init"])
    model2.X = Param(model.O, model.N, initialize=plsobj_["pyo_X"])
    model2.theta = Param(model.O, model.M, initialize=plsobj_["pyo_theta"])
    model2.Y = Param(model.O, model.M, initialize=plsobj_["pyo_Y"])
    model2.delta = Param(
        model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1 == a2 else 0
    )

    def _eq_36a_mod_obj(model):
        return sum(
            sum(
                sum(
                    model.X[o, n]
                    * (model.Y[o, m] - model.theta[o, m] * sum(model.T[o, a] * model.Q[m, a] for a in model.A))
                    for o in model.O
                ) ** 2
                for n in model.N
            )
            for m in model.M
        )
    model2.obj = Objective(rule=_eq_36a_mod_obj)

    if ipopt_ok:
        solver = SolverFactory("ipopt")
        results = solver.solve(model2, tee=True)
    elif gams_ok:
        solver = SolverFactory("gams:ipopt")
        results = solver.solve(model2, tee=True)
    else:
        solver_manager = SolverManagerFactory("neos")
        results = solver_manager.solve(model2, opt="ipopt", tee=True)

    Q = np.array([[value(model2.Q[m, a]) for a in model2.A] for m in model2.M])

    # Calculate R²
    for a in range(A):
        ti = T[:, [a]]
        pi = P[:, [a]]
        qi = Q[:, [a]]
        X_ = (X_ - ti @ pi.T) * not_Xmiss
        Y_ = (Y_ - ti @ qi.T) * not_Ymiss
        if a == 0:
            r2X = 1 - np.sum(X_**2) / TSSX
            r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
            r2Y = 1 - np.sum(Y_**2) / TSSY
            r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
        else:
            r2X_ = 1 - np.sum(X_**2) / TSSX
            r2Xpv_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
            r2X = np.hstack((r2X, r2X_))
            r2Xpv = np.hstack((r2Xpv, r2Xpv_))
            r2Y_ = 1 - np.sum(Y_**2) / TSSY
            r2Ypv_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
            r2Y = np.hstack((r2Y, r2Y_))
            r2Ypv = np.hstack((r2Ypv, r2Ypv_))

    for a in range(A - 1, 0, -1):
        r2X[a] = r2X[a] - r2X[a - 1]
        r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]

    eigs = np.var(T, axis=0)
    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)

    if not shush:
        print("-" * 62)
        print("LV #     Eig       R2X       sum(R2X)   R2Y       sum(R2Y)")
        if A > 1:
            for a in range(A):
                print(
                    f"LV #{a + 1}:   {eigs[a]:6.3f}    {r2X[a]:.3f}     "
                    f"{r2xc[a]:.3f}      {r2Y[a]:.3f}     {r2yc[a]:.3f}"
                )
        else:
            print(
                f"LV #1:   {eigs[0]:6.3f}    {r2X:.3f}     "
                f"{r2xc[0]:.3f}      {r2Y:.3f}     {r2yc[0]:.3f}"
            )
        print("-" * 62)

    W = 1  # Not computed in NLP
    U = 1  # Not computed in NLP

    pls_obj = {
        "T": T, "P": P, "Q": Q, "W": W, "Ws": Ws, "U": U,
        "r2x": r2X, "r2xpv": r2Xpv, "mx": x_mean, "sx": x_std,
        "r2y": r2Y, "r2ypv": r2Ypv, "my": y_mean, "sy": y_std,
    }

    if not isinstance(obsidX, bool):
        pls_obj["obsidX"] = obsidX
        pls_obj["varidX"] = varidX
    if not isinstance(obsidY, bool):
        pls_obj["obsidY"] = obsidY
        pls_obj["varidY"] = varidY

    # Add diagnostics
    _add_pls_diagnostics(pls_obj, T, X_, Y_, A)

    return pls_obj


def _add_pls_diagnostics(
    pls_obj: dict,
    T: np.ndarray,
    X_residual: np.ndarray,
    Y_residual: np.ndarray,
    A: int,
) -> None:
    """Add diagnostic statistics to PLS object."""
    T2 = hott2(pls_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, n - A)
    T2_lim95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, n - A)

    speX = np.sum(X_residual**2, axis=1, keepdims=True)
    speX_lim95, speX_lim99 = spe_ci(speX)
    speY = np.sum(Y_residual**2, axis=1, keepdims=True)
    speY_lim95, speY_lim99 = spe_ci(speY)

    pls_obj["T2"] = T2
    pls_obj["T2_lim99"] = T2_lim99
    pls_obj["T2_lim95"] = T2_lim95
    pls_obj["speX"] = speX
    pls_obj["speX_lim99"] = speX_lim99
    pls_obj["speX_lim95"] = speX_lim95
    pls_obj["speY"] = speY
    pls_obj["speY_lim99"] = speY_lim99
    pls_obj["speY_lim95"] = speY_lim95


# =============================================================================
# Main PLS Function (with cross-validation)
# =============================================================================


def pls(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcsX: bool | str = True,
    mcsY: bool | str = True,
    md_algorithm: str = "nipals",
    force_nipals: bool = True,
    shush: bool = False,
    cross_val: int = 0,
    cross_val_X: bool = False,
    cca: bool = False,
) -> dict:
    """Create a Projection to Latent Structures (PLS) model.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Predictor data. If DataFrame, first column is observation IDs.
    Y : np.ndarray or pd.DataFrame
        Response data. If DataFrame, first column is observation IDs.
    A : int
        Number of Latent Variables to calculate.
    mcsX : bool or str, default=True
        X preprocessing: True (center+scale), False (none), 'center', 'autoscale'.
    mcsY : bool or str, default=True
        Y preprocessing: True (center+scale), False (none), 'center', 'autoscale'.
    md_algorithm : str, default="nipals"
        Missing data algorithm ("nipals" or "nlp").
    force_nipals : bool, default=True
        Force NIPALS even for complete data.
    shush : bool, default=False
        Suppress output.
    cross_val : int, default=0
        Cross-validation percentage (0-100). 0=skip, 100=leave-one-out.
    cross_val_X : bool, default=False
        If True, also calculate Q² for X matrix.
    cca : bool, default=False
        Calculate CCA for OPLS-like covariant scores/loadings.

    Returns
    -------
    dict
        PLS model object containing:
        - T, P, Q, W, Ws, U: Model matrices
        - r2x, r2y, r2xpv, r2ypv: R² statistics
        - mx, sx, my, sy: Preprocessing parameters
        - T2, speX, speY: Diagnostics with confidence limits
        - q2Y, q2Ypv: (if cross_val > 0) Q² for Y
        - q2X, q2Xpv: (if cross_val > 0 and cross_val_X) Q² for X
        - Tcv, Pcv, Wcv, Betacv: (if cca=True) CCA results
        - type: "pls"

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 10)
    >>> Y = np.random.randn(50, 2)
    >>> model = pls(X, Y, 3)
    >>> model['T'].shape
    (50, 3)
    """
    if cross_val == 0:
        plsobj = pls_(
            X, Y, A,
            mcsX=mcsX, mcsY=mcsY,
            md_algorithm=md_algorithm,
            force_nipals=force_nipals,
            shush=shush,
            cca_flag=cca,
        )
        plsobj["type"] = "pls"
    elif 0 < cross_val <= 100:
        plsobj = _pls_cross_validate(
            X, Y, A,
            mcsX=mcsX, mcsY=mcsY,
            cross_val=cross_val,
            cross_val_X=cross_val_X,
            shush=shush,
            cca_flag=cca,
        )
    else:
        plsobj = "Cannot cross validate with those options"
    return plsobj


def _pls_cross_validate(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcsX: bool | str = True,
    mcsY: bool | str = True,
    cross_val: int = 10,
    cross_val_X: bool = False,
    shush: bool = False,
    cca_flag: bool = False,
) -> dict:
    """Internal function for PLS with cross-validation."""
    # Extract data
    if isinstance(X, np.ndarray):
        X_ = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)

    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    elif isinstance(Y, pd.DataFrame):
        Y_ = np.array(Y.values[:, 1:]).astype(float)

    # Preprocess X
    if isinstance(mcsX, bool):
        if mcsX:
            X_, x_mean, x_std = meancenterscale(X_)
        else:
            x_mean = np.zeros((1, X_.shape[1]))
            x_std = np.ones((1, X_.shape[1]))
    elif mcsX == "center":
        X_, x_mean, x_std = meancenterscale(X_, mcs="center")
    elif mcsX == "autoscale":
        X_, x_mean, x_std = meancenterscale(X_, mcs="autoscale")

    # Preprocess Y
    if isinstance(mcsY, bool):
        if mcsY:
            Y_, y_mean, y_std = meancenterscale(Y_)
        else:
            y_mean = np.zeros((1, Y_.shape[1]))
            y_std = np.ones((1, Y_.shape[1]))
    elif mcsY == "center":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="center")
    elif mcsY == "autoscale":
        Y_, y_mean, y_std = meancenterscale(Y_, mcs="autoscale")

    # Generate missing data maps
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map).astype(int)
    Y_nan_map = np.isnan(Y_)
    not_Ymiss = (~Y_nan_map).astype(int)

    # Initialize TSS
    X_, Xnanmap = n2z(X_.copy())
    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)
    colsX = X_.shape[1]
    rowsX = X_.shape[0]
    X_ = z2n(X_, Xnanmap)

    Y_, Ynanmap = n2z(Y_.copy())
    TSSY = np.sum(Y_**2)
    TSSYpv = np.sum(Y_**2, axis=0)
    colsY = Y_.shape[1]
    rowsY = Y_.shape[0]
    Y_ = z2n(Y_, Ynanmap)

    # Leave-one-out cross-validation
    if cross_val == 100:
        for a in range(A):
            errorY = np.zeros((rowsY * colsY, 1))
            if cross_val_X:
                errorX = np.zeros((rowsX * colsX, 1))

            for o in range(X_.shape[0]):
                X_copy = X_.copy()
                Y_copy = Y_.copy()

                elements_outX = X_copy[o, :].copy()
                elements_outY = Y_copy[o, :].copy()
                X_copy = np.delete(X_copy, o, 0)
                Y_copy = np.delete(Y_copy, o, 0)

                plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
                plspred = pls_pred(elements_outX, plsobj_)

                if o == 0:
                    if cross_val_X:
                        errorX = elements_outX - plspred["Xhat"]
                    errorY = elements_outY - plspred["Yhat"]
                else:
                    if cross_val_X:
                        errorX = np.vstack((errorX, elements_outX - plspred["Xhat"]))
                    errorY = np.vstack((errorY, elements_outY - plspred["Yhat"]))

            if cross_val_X:
                errorX, dummy = n2z(errorX.copy())
                PRESSXpv = np.sum(errorX**2, axis=0)
                PRESSX = np.sum(errorX**2)

            errorY, dummy = n2z(errorY.copy())
            PRESSYpv = np.sum(errorY**2, axis=0)
            PRESSY = np.sum(errorY**2)

            if a == 0:
                q2Y = 1 - PRESSY / TSSY
                q2Ypv = (1 - PRESSYpv / TSSYpv).reshape(-1, 1)
                if cross_val_X:
                    q2X = 1 - PRESSX / TSSX
                    q2Xpv = (1 - PRESSXpv / TSSXpv).reshape(-1, 1)
            else:
                q2Y = np.hstack((q2Y, 1 - PRESSY / TSSY))
                aux_ = (1 - PRESSYpv / TSSYpv).reshape(-1, 1)
                q2Ypv = np.hstack((q2Ypv, aux_))
                if cross_val_X:
                    q2X = np.hstack((q2X, 1 - PRESSX / TSSX))
                    aux_ = (1 - PRESSXpv / TSSXpv).reshape(-1, 1)
                    q2Xpv = np.hstack((q2Xpv, aux_))

            # Deflate
            X_copy = X_.copy()
            Y_copy = Y_.copy()
            plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
            xhat = plsobj_["T"] @ plsobj_["P"].T
            yhat = plsobj_["T"] @ plsobj_["Q"].T
            X_, Xnanmap = n2z(X_.copy())
            Y_, Ynanmap = n2z(Y_.copy())
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss

            if a == 0:
                r2X = 1 - np.sum(X_**2) / TSSX
                r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                r2Y = 1 - np.sum(Y_**2) / TSSY
                r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
            else:
                r2X = np.hstack((r2X, 1 - np.sum(X_**2) / TSSX))
                aux_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                r2Xpv = np.hstack((r2Xpv, aux_))
                r2Y = np.hstack((r2Y, 1 - np.sum(Y_**2) / TSSY))
                aux_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                r2Ypv = np.hstack((r2Ypv, aux_))

            X_ = z2n(X_, Xnanmap)
            Y_ = z2n(Y_, Ynanmap)

    else:  # Element-wise cross-validation
        for a in range(A):
            if not shush:
                print(f"Cross validating LV #{a + 1}")

            # Generate cross-val map for Y
            not_removed_mapY = not_Ymiss.copy()
            not_removed_mapY = np.reshape(not_removed_mapY, (rowsY * colsY, -1))
            Yrnd = np.random.random(Y_.shape) * not_Ymiss
            indxY = np.argsort(np.reshape(Yrnd, (Yrnd.shape[0] * Yrnd.shape[1])))
            elements_to_remove_per_roundY = int(
                np.ceil((Y_.shape[0] * Y_.shape[1]) * (cross_val / 100))
            )
            errorY = np.zeros((rowsY * colsY, 1))

            if cross_val_X:
                not_removed_mapX = not_Xmiss.copy()
                not_removed_mapX = np.reshape(not_removed_mapX, (rowsX * colsX, -1))
                Xrnd = np.random.random(X_.shape) * not_Xmiss
                indxX = np.argsort(np.reshape(Xrnd, (Xrnd.shape[0] * Xrnd.shape[1])))
                elements_to_remove_per_roundX = int(
                    np.ceil((X_.shape[0] * X_.shape[1]) * (cross_val / 100))
                )
                errorX = np.zeros((rowsX * colsX, 1))
            else:
                not_removed_mapX = 0

            while np.sum(not_removed_mapX) > 0 or np.sum(not_removed_mapY) > 0:
                X_copy = X_.copy()
                if cross_val_X:
                    if indxX.size > elements_to_remove_per_roundX:
                        indx_this_roundX = indxX[:elements_to_remove_per_roundX]
                        indxX = indxX[elements_to_remove_per_roundX:]
                    else:
                        indx_this_roundX = indxX
                    X_copy = np.reshape(X_copy, (rowsX * colsX, 1))
                    elements_outX = X_copy[indx_this_roundX]
                    X_copy[indx_this_roundX] = np.nan
                    X_copy = np.reshape(X_copy, (rowsX, colsX))
                    not_removed_mapX[indx_this_roundX] = 0
                    auxmap = np.isnan(X_copy).astype(int)
                    auxmap = np.sum(auxmap, axis=1)
                    indx2 = np.where(auxmap == X_copy.shape[1])[0].tolist()
                else:
                    indx2 = []

                Y_copy = Y_.copy()
                if indxY.size > elements_to_remove_per_roundY:
                    indx_this_roundY = indxY[:elements_to_remove_per_roundY]
                    indxY = indxY[elements_to_remove_per_roundY:]
                else:
                    indx_this_roundY = indxY
                Y_copy = np.reshape(Y_copy, (rowsY * colsY, 1))
                elements_outY = Y_copy[indx_this_roundY]
                Y_copy[indx_this_roundY] = np.nan
                Y_copy = np.reshape(Y_copy, (rowsY, colsY))
                not_removed_mapY[indx_this_roundY] = 0
                auxmap = np.isnan(Y_copy).astype(int)
                auxmap = np.sum(auxmap, axis=1)
                indx3 = np.where(auxmap == Y_copy.shape[1])[0].tolist()
                indx4 = np.unique(indx3 + indx2).tolist()

                if len(indx4) > 0:
                    X_copy = np.delete(X_copy, indx4, 0)
                    Y_copy = np.delete(Y_copy, indx4, 0)

                plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
                plspred = pls_pred(X_, plsobj_)

                if cross_val_X:
                    xhat = plspred["Tnew"] @ plsobj_["P"].T
                    xhat = np.reshape(xhat, (rowsX * colsX, 1))
                    errorX[indx_this_roundX] = elements_outX - xhat[indx_this_roundX]

                yhat = plspred["Tnew"] @ plsobj_["Q"].T
                yhat = np.reshape(yhat, (rowsY * colsY, 1))
                errorY[indx_this_roundY] = elements_outY - yhat[indx_this_roundY]

            if cross_val_X:
                errorX = np.reshape(errorX, (rowsX, colsX))
                errorX, dummy = n2z(errorX.copy())
                PRESSXpv = np.sum(errorX**2, axis=0)
                PRESSX = np.sum(errorX**2)

            errorY = np.reshape(errorY, (rowsY, colsY))
            errorY, dummy = n2z(errorY.copy())
            PRESSYpv = np.sum(errorY**2, axis=0)
            PRESSY = np.sum(errorY**2)

            if a == 0:
                q2Y = 1 - PRESSY / TSSY
                q2Ypv = (1 - PRESSYpv / TSSYpv).reshape(-1, 1)
                if cross_val_X:
                    q2X = 1 - PRESSX / TSSX
                    q2Xpv = (1 - PRESSXpv / TSSXpv).reshape(-1, 1)
            else:
                q2Y = np.hstack((q2Y, 1 - PRESSY / TSSY))
                aux_ = (1 - PRESSYpv / TSSYpv).reshape(-1, 1)
                q2Ypv = np.hstack((q2Ypv, aux_))
                if cross_val_X:
                    q2X = np.hstack((q2X, 1 - PRESSX / TSSX))
                    aux_ = (1 - PRESSXpv / TSSXpv).reshape(-1, 1)
                    q2Xpv = np.hstack((q2Xpv, aux_))

            # Deflate
            X_copy = X_.copy()
            Y_copy = Y_.copy()
            plsobj_ = pls_(X_copy, Y_copy, 1, mcsX=False, mcsY=False, shush=True)
            xhat = plsobj_["T"] @ plsobj_["P"].T
            yhat = plsobj_["T"] @ plsobj_["Q"].T
            X_, Xnanmap = n2z(X_.copy())
            Y_, Ynanmap = n2z(Y_.copy())
            X_ = (X_ - xhat) * not_Xmiss
            Y_ = (Y_ - yhat) * not_Ymiss

            if a == 0:
                r2X = 1 - np.sum(X_**2) / TSSX
                r2Xpv = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                r2Y = 1 - np.sum(Y_**2) / TSSY
                r2Ypv = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
            else:
                r2X = np.hstack((r2X, 1 - np.sum(X_**2) / TSSX))
                aux_ = (1 - np.sum(X_**2, axis=0) / TSSXpv).reshape(-1, 1)
                r2Xpv = np.hstack((r2Xpv, aux_))
                r2Y = np.hstack((r2Y, 1 - np.sum(Y_**2) / TSSY))
                aux_ = (1 - np.sum(Y_**2, axis=0) / TSSYpv).reshape(-1, 1)
                r2Ypv = np.hstack((r2Ypv, aux_))

            X_ = z2n(X_, Xnanmap)
            Y_ = z2n(Y_, Ynanmap)

    # Fit full model
    plsobj = pls_(X, Y, A, mcsX=mcsX, mcsY=mcsY, shush=True, cca_flag=cca_flag)

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2X[a] = r2X[a] - r2X[a - 1]
        r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
        if cross_val_X:
            q2X[a] = q2X[a] - q2X[a - 1]
            q2Xpv[:, a] = q2Xpv[:, a] - q2Xpv[:, a - 1]
        else:
            q2X = False
            q2Xpv = False
        r2Y[a] = r2Y[a] - r2Y[a - 1]
        r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]
        q2Y[a] = q2Y[a] - q2Y[a - 1]
        q2Ypv[:, a] = q2Ypv[:, a] - q2Ypv[:, a - 1]

    r2xc = np.cumsum(r2X)
    r2yc = np.cumsum(r2Y)
    if cross_val_X:
        q2xc = np.cumsum(q2X)
    else:
        q2xc = False
    q2yc = np.cumsum(q2Y)
    eigs = np.var(plsobj["T"], axis=0)

    plsobj["q2Y"] = q2Y
    plsobj["q2Ypv"] = q2Ypv
    if cross_val_X:
        plsobj["q2X"] = q2X
        plsobj["q2Xpv"] = q2Xpv

    if not shush:
        print(
            f"phi.pls using NIPALS and cross-validation ({cross_val}%) "
            f"executed on: {datetime.datetime.now()}"
        )
        if not cross_val_X:
            print("-" * 81)
            print("PC #       Eig      R2X     sum(R2X)      R2Y     sum(R2Y)      Q2Y     sum(Q2Y)")
            if A > 1:
                for a in range(A):
                    print(
                        f"PC #{a + 1}:{eigs[a]:8.3f}    {r2X[a]:.3f}     {r2xc[a]:.3f}       "
                        f"{r2Y[a]:.3f}     {r2yc[a]:.3f}       {q2Y[a]:.3f}     {q2yc[a]:.3f}"
                    )
            else:
                print(
                    f"PC #1:{eigs[0]:8.3f}    {r2X:.3f}     {r2xc[0]:.3f}       "
                    f"{r2Y:.3f}     {r2yc[0]:.3f}       {q2Y:.3f}     {q2yc[0]:.3f}"
                )
            print("-" * 81)
        else:
            print("-" * 103)
            print(
                "PC #       Eig      R2X     sum(R2X)      Q2X     sum(Q2X)      "
                "R2Y     sum(R2Y)      Q2Y     sum(Q2Y)"
            )
            if A > 1:
                for a in range(A):
                    print(
                        f"PC #{a + 1}:{eigs[a]:8.3f}    {r2X[a]:.3f}     {r2xc[a]:.3f}       "
                        f"{q2X[a]:.3f}     {q2xc[a]:.3f}       {r2Y[a]:.3f}     {r2yc[a]:.3f}       "
                        f"{q2Y[a]:.3f}     {q2yc[a]:.3f}"
                    )
            else:
                print(
                    f"PC #1:{eigs[0]:8.3f}    {r2X:.3f}     {r2xc[0]:.3f}       "
                    f"{q2X:.3f}     {q2xc[0]:.3f}       {r2Y:.3f}     {r2yc[0]:.3f}       "
                    f"{q2Y:.3f}     {q2yc[0]:.3f}"
                )
            print("-" * 103)

    plsobj["type"] = "pls"
    return plsobj

