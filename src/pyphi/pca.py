"""
Principal Component Analysis (PCA) module for PyPhi.

This module implements PCA with support for:
- SVD algorithm for complete data
- NIPALS algorithm for data with missing values
- NLP optimization for missing data (via Pyomo)
- Cross-validation for model selection

References:
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


def hott2(
    mvmobj: dict,
    *,
    Xnew: np.ndarray | bool = False,
    Tnew: np.ndarray | bool = False,
) -> np.ndarray:
    """Calculate Hotelling's T² statistic.

    Parameters
    ----------
    mvmobj : dict
        PCA or PLS model object containing 'T' (scores matrix).
    Xnew : np.ndarray or False, default=False
        New X data to project and calculate T².
    Tnew : np.ndarray or False, default=False
        Pre-calculated scores for new data.

    Returns
    -------
    np.ndarray
        Hotelling's T² values for each observation.

    Notes
    -----
    At least one of Xnew or Tnew should be provided. If neither is provided,
    T² is calculated for the training data scores in mvmobj['T'].
    """
    if isinstance(Xnew, bool) and not isinstance(Tnew, bool):
        # Use provided Tnew
        var_t = (mvmobj["T"].T @ mvmobj["T"]) / mvmobj["T"].shape[0]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew, axis=1)
    elif isinstance(Tnew, bool) and not isinstance(Xnew, bool):
        # Project Xnew to get Tnew
        if "Q" in mvmobj:
            # PLS model - would need pls_pred
            raise NotImplementedError("Use pls_pred for PLS models")
        else:
            # PCA model
            xpred = pca_pred(Xnew, mvmobj)
        Tnew = xpred["Tnew"]
        var_t = (mvmobj["T"].T @ mvmobj["T"]) / mvmobj["T"].shape[0]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew, axis=1)
    elif isinstance(Xnew, bool) and isinstance(Tnew, bool):
        # Use training data
        var_t = (mvmobj["T"].T @ mvmobj["T"]) / mvmobj["T"].shape[0]
        Tnew = mvmobj["T"]
        hott2_ = np.sum((Tnew @ np.linalg.inv(var_t)) * Tnew, axis=1)
    return hott2_


def prep_pca_4_MDbyNLP(pcaobj: dict, X: np.ndarray) -> dict:
    """Prepare PCA object for missing data handling via NLP.

    Converts PCA model parameters to Pyomo-compatible format for
    the NLP optimization approach.

    Parameters
    ----------
    pcaobj : dict
        PCA model object from pca_().
    X : np.ndarray
        Data matrix (may contain NaN).

    Returns
    -------
    dict
        Augmented PCA object with Pyomo-formatted parameters.
    """
    pcaobj_ = pcaobj.copy()
    X_nan_map = np.isnan(X)
    psi = (~X_nan_map).astype(int)
    X, dummy = n2z(X.copy())

    A = pcaobj["T"].shape[1]
    O = pcaobj["T"].shape[0]
    N = pcaobj["P"].shape[0]

    pyo_A = list(range(1, A + 1))
    pyo_N = list(range(1, N + 1))
    pyo_O = list(range(1, O + 1))

    pyo_P_init = np2D2pyomo(pcaobj["P"])
    pyo_T_init = np2D2pyomo(pcaobj["T"])
    pyo_X = np2D2pyomo(X)
    pyo_psi = np2D2pyomo(psi)

    pcaobj_["pyo_A"] = pyo_A
    pcaobj_["pyo_N"] = pyo_N
    pcaobj_["pyo_O"] = pyo_O
    pcaobj_["pyo_P_init"] = pyo_P_init
    pcaobj_["pyo_T_init"] = pyo_T_init
    pcaobj_["pyo_X"] = pyo_X
    pcaobj_["pyo_psi"] = pyo_psi

    return pcaobj_


def pca(
    X: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcs: bool | str = True,
    md_algorithm: str = "nipals",
    force_nipals: bool = False,
    shush: bool = False,
    cross_val: int = 0,
) -> dict:
    """Create a Principal Components Analysis model.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data to train the model. If DataFrame, first column is observation IDs.
    A : int
        Number of Principal Components to calculate.
    mcs : bool or str, default=True
        Preprocessing mode:
        - True: Mean-center and autoscale
        - False: No preprocessing
        - "center": Only center
        - "autoscale": Only autoscale
    md_algorithm : str, default="nipals"
        Missing data algorithm:
        - "nipals": NIPALS algorithm
        - "nlp": Non-linear programming (Lopez-Negrete et al.)
    force_nipals : bool, default=False
        If True, use NIPALS even for complete data. If False, uses SVD
        when data is complete and matrix is tall/wide enough.
    shush : bool, default=False
        If True, suppress printed output.
    cross_val : int, default=0
        Cross-validation percentage (0-100). If 0, skip cross-validation.

    Returns
    -------
    dict
        PCA model object containing:
        - T: Scores matrix
        - P: Loadings matrix
        - r2x: R² per component
        - r2xpv: R² per variable per component
        - mx, sx: Mean and std used for preprocessing
        - T2, T2_lim95, T2_lim99: Hotelling's T² and limits
        - speX, speX_lim95, speX_lim99: SPE and limits
        - q2, q2pv: (if cross_val > 0) Cross-validated R²
        - obsidX, varidX: (if DataFrame input) Observation and variable IDs
        - type: "pca"

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(50, 10)
    >>> model = pca(X, 3)
    >>> model['T'].shape
    (50, 3)
    """
    if cross_val == 0:
        pcaobj = pca_(
            X,
            A,
            mcs=mcs,
            md_algorithm=md_algorithm,
            force_nipals=force_nipals,
            shush=shush,
        )
        pcaobj["type"] = "pca"
    elif 0 < cross_val < 100:
        pcaobj = _pca_cross_validate(
            X, A, mcs=mcs, cross_val=cross_val, shush=shush
        )
    else:
        pcaobj = "Cannot cross validate with those options"
    return pcaobj


def _pca_cross_validate(
    X: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcs: bool | str = True,
    cross_val: int = 10,
    shush: bool = False,
) -> dict:
    """Internal function for PCA with cross-validation."""
    if isinstance(X, np.ndarray):
        X_ = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)

    # Preprocess
    if isinstance(mcs, bool):
        if mcs:
            X_, x_mean, x_std = meancenterscale(X_)
        else:
            x_mean = np.zeros((1, X_.shape[1]))
            x_std = np.ones((1, X_.shape[1]))
    elif mcs == "center":
        X_, x_mean, x_std = meancenterscale(X_, mcs="center")
    elif mcs == "autoscale":
        X_, x_mean, x_std = meancenterscale(X_, mcs="autoscale")

    # Generate missing data map
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map).astype(int)

    # Initialize TSS
    X_, Xnanmap = n2z(X_.copy())
    TSS = np.sum(X_**2)
    TSSpv = np.sum(X_**2, axis=0)
    cols = X_.shape[1]
    rows = X_.shape[0]
    X_ = z2n(X_, Xnanmap)

    for a in range(A):
        if not shush:
            print(f"Cross validating PC #{a + 1}")

        # Generate cross-val map
        not_removed_map = not_Xmiss.copy()
        not_removed_map = np.reshape(not_removed_map, (rows * cols, -1))

        # Generate random removal order
        Xrnd = np.random.random(X_.shape) * not_Xmiss
        indx = np.argsort(np.reshape(Xrnd, (Xrnd.shape[0] * Xrnd.shape[1])))
        elements_to_remove = int(
            np.ceil((X_.shape[0] * X_.shape[1]) * (cross_val / 100))
        )
        error = np.zeros((rows * cols, 1))
        rounds = 1

        while np.sum(not_removed_map) > 0:
            rounds += 1
            X_copy = X_.copy()
            if indx.size > elements_to_remove:
                indx_this_round = indx[:elements_to_remove]
                indx = indx[elements_to_remove:]
            else:
                indx_this_round = indx

            # Place NaN's
            X_copy = np.reshape(X_copy, (rows * cols, 1))
            elements_out = X_copy[indx_this_round]
            X_copy[indx_this_round] = np.nan
            X_copy = np.reshape(X_copy, (rows, cols))

            # Update map
            not_removed_map[indx_this_round] = 0

            # Look for rows with all missing data
            auxmap = np.isnan(X_copy).astype(int)
            auxmap = np.sum(auxmap, axis=1)
            indx2 = np.where(auxmap == X_copy.shape[1])[0].tolist()
            if len(indx2) > 0:
                X_copy = np.delete(X_copy, indx2, 0)

            pcaobj_ = pca_(X_copy, 1, mcs=False, shush=True)
            xhat = pcaobj_["T"] @ pcaobj_["P"].T
            xhat = np.insert(xhat, indx2, np.nan, axis=0)
            xhat = np.reshape(xhat, (rows * cols, 1))
            error[indx_this_round] = elements_out - xhat[indx_this_round]

        error = np.reshape(error, (rows, cols))
        error, dummy = n2z(error.copy())
        PRESSpv = np.sum(error**2, axis=0)
        PRESS = np.sum(error**2)

        if a == 0:
            q2 = 1 - PRESS / TSS
            q2pv = (1 - PRESSpv / TSSpv).reshape(-1, 1)
        else:
            q2 = np.hstack((q2, 1 - PRESS / TSS))
            aux_ = (1 - PRESSpv / TSSpv).reshape(-1, 1)
            q2pv = np.hstack((q2pv, aux_))

        # Deflate and go to next PC
        X_copy = X_.copy()
        pcaobj_ = pca_(X_copy, 1, mcs=False, shush=True)
        xhat = pcaobj_["T"] @ pcaobj_["P"].T
        X_, Xnanmap = n2z(X_.copy())
        X_ = (X_ - xhat) * not_Xmiss

        if a == 0:
            r2 = 1 - np.sum(X_**2) / TSS
            r2pv = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
        else:
            r2 = np.hstack((r2, 1 - np.sum(X_**2) / TSS))
            aux_ = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
            r2pv = np.hstack((r2pv, aux_))

        X_ = z2n(X_, Xnanmap)

    # Fit full model
    pcaobj = pca_(X, A, mcs=mcs, force_nipals=True, shush=True)

    # Convert cumulative R² to per-component
    for a in range(A - 1, 0, -1):
        r2[a] = r2[a] - r2[a - 1]
        r2pv[:, a] = r2pv[:, a] - r2pv[:, a - 1]
        q2[a] = q2[a] - q2[a - 1]
        q2pv[:, a] = q2pv[:, a] - q2pv[:, a - 1]

    r2xc = np.cumsum(r2)
    q2xc = np.cumsum(q2)
    eigs = np.var(pcaobj["T"], axis=0)
    pcaobj["q2"] = q2
    pcaobj["q2pv"] = q2pv

    if not shush:
        print(
            f"phi.pca using NIPALS and cross validation ({cross_val}%) "
            f"executed on: {datetime.datetime.now()}"
        )
        print("-" * 62)
        print("PC #          Eig      R2X     sum(R2X)      Q2X     sum(Q2X)")
        if A > 1:
            for a in range(A):
                print(
                    f"PC #{a + 1}:   {eigs[a]:8.3f}    {r2[a]:.3f}     "
                    f"{r2xc[a]:.3f}       {q2[a]:.3f}     {q2xc[a]:.3f}"
                )
        else:
            print(
                f"PC #1:   {eigs[0]:8.3f}    {r2:.3f}     "
                f"{r2xc[0]:.3f}       {q2:.3f}     {q2xc[0]:.3f}"
            )
        print("-" * 62)

    pcaobj["type"] = "pca"
    return pcaobj


def pca_(
    X: np.ndarray | pd.DataFrame,
    A: int,
    *,
    mcs: bool | str = True,
    md_algorithm: str = "nipals",
    force_nipals: bool = False,
    shush: bool = False,
) -> dict:
    """Core PCA algorithm implementation.

    This is the internal function that implements the actual PCA computation.
    Use `pca()` for the public interface with cross-validation support.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data matrix.
    A : int
        Number of components.
    mcs : bool or str, default=True
        Preprocessing mode.
    md_algorithm : str, default="nipals"
        Missing data algorithm ("nipals" or "nlp").
    force_nipals : bool, default=False
        Force NIPALS even for complete data.
    shush : bool, default=False
        Suppress output.

    Returns
    -------
    dict
        PCA model object.
    """
    # Extract data and identifiers
    if isinstance(X, np.ndarray):
        X_ = X.copy()
        obsidX = False
        varidX = False
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)
        obsidX = X.values[:, 0].astype(str).tolist()
        varidX = X.columns.values[1:].tolist()

    # Preprocess
    if isinstance(mcs, bool):
        if mcs:
            X_, x_mean, x_std = meancenterscale(X_)
        else:
            x_mean = np.zeros((1, X_.shape[1]))
            x_std = np.ones((1, X_.shape[1]))
    elif mcs == "center":
        X_, x_mean, x_std = meancenterscale(X_, mcs="center")
    elif mcs == "autoscale":
        X_, x_mean, x_std = meancenterscale(X_, mcs="autoscale")

    # Generate missing data map
    X_nan_map = np.isnan(X_)
    not_Xmiss = (~X_nan_map).astype(int)

    # Choose algorithm based on data completeness and shape
    use_svd = (
        not X_nan_map.any()
        and not force_nipals
        and (X_.shape[1] / X_.shape[0] >= 10 or X_.shape[0] / X_.shape[1] >= 10)
    )

    if use_svd:
        return _pca_svd(X_, A, x_mean, x_std, obsidX, varidX, shush)
    elif md_algorithm == "nipals":
        return _pca_nipals(X_, A, x_mean, x_std, not_Xmiss, obsidX, varidX, shush)
    elif md_algorithm == "nlp" and _PYOMO_AVAILABLE:
        return _pca_nlp(X, X_, A, mcs, x_mean, x_std, not_Xmiss, obsidX, varidX, shush)
    elif md_algorithm == "nlp" and not _PYOMO_AVAILABLE:
        print("Pyomo was not found in your system")
        print("visit http://www.pyomo.org/")
        return {"error": "Pyomo not available"}


def _pca_svd(
    X_: np.ndarray,
    A: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    shush: bool,
) -> dict:
    """PCA using SVD for complete data."""
    if not shush:
        print(f"phi.pca using SVD executed on: {datetime.datetime.now()}")

    TSS = np.sum(X_**2)
    TSSpv = np.sum(X_**2, axis=0)

    if X_.shape[1] / X_.shape[0] >= 10:
        # Wide matrix
        U, S, Th = np.linalg.svd(X_ @ X_.T)
        T = Th.T[:, :A]
        P = X_.T @ T
        for a in range(A):
            P[:, a] = P[:, a] / np.linalg.norm(P[:, a])
        T = X_ @ P
    elif X_.shape[0] / X_.shape[1] >= 10:
        # Tall matrix
        U, S, Ph = np.linalg.svd(X_.T @ X_)
        P = Ph.T[:, :A]
        T = X_ @ P

    # Calculate R²
    X_deflated = X_.copy()
    for a in range(A):
        X_deflated = X_deflated - T[:, [a]] @ P[:, [a]].T
        if a == 0:
            r2 = 1 - np.sum(X_deflated**2) / TSS
            r2pv = (1 - np.sum(X_deflated**2, axis=0) / TSSpv).reshape(-1, 1)
        else:
            r2 = np.hstack((r2, 1 - np.sum(X_deflated**2) / TSS))
            aux_ = (1 - np.sum(X_deflated**2, axis=0) / TSSpv).reshape(-1, 1)
            r2pv = np.hstack((r2pv, aux_))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2[a] = r2[a] - r2[a - 1]
        r2pv[:, a] = r2pv[:, a] - r2pv[:, a - 1]

    pca_obj = {"T": T, "P": P, "r2x": r2, "r2xpv": r2pv, "mx": x_mean, "sx": x_std}
    if not isinstance(obsidX, bool):
        pca_obj["obsidX"] = obsidX
        pca_obj["varidX"] = varidX

    # Add diagnostics
    _add_pca_diagnostics(pca_obj, T, X_deflated, A, shush)

    return pca_obj


def _pca_nipals(
    X_: np.ndarray,
    A: int,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    not_Xmiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    shush: bool,
) -> dict:
    """PCA using NIPALS algorithm (handles missing data)."""
    if not shush:
        print(f"phi.pca using NIPALS executed on: {datetime.datetime.now()}")

    X_, dummy = n2z(X_.copy())
    epsilon = 1e-10
    maxit = 5000
    TSS = np.sum(X_**2)
    TSSpv = np.sum(X_**2, axis=0)

    for a in range(A):
        # Initial guess: column with largest variance
        ti = X_[:, [np.argmax(std(X_))]]
        converged = False
        num_it = 0

        while not converged:
            # Step 1: p = t'X / t't
            timat = np.tile(ti, (1, X_.shape[1]))
            pi = np.sum(X_ * timat, axis=0) / np.sum((timat * not_Xmiss) ** 2, axis=0)

            # Step 2: Normalize p
            pi = pi / np.linalg.norm(pi)

            # Step 3: t_new = Xp / p'p
            pimat = np.tile(pi, (X_.shape[0], 1))
            tn = X_ @ pi.T
            ptp = np.sum((pimat * not_Xmiss) ** 2, axis=1)
            tn = tn / ptp
            pi = pi.reshape(-1, 1)

            # Check convergence
            if abs(np.linalg.norm(ti) - np.linalg.norm(tn)) / np.linalg.norm(ti) < epsilon:
                converged = True
            if num_it > maxit:
                converged = True

            if converged:
                # Rotate so variance is larger for positive scores
                if len(ti[ti < 0]) > 0 and len(ti[ti > 0]) > 0:
                    if np.var(ti[ti < 0]) > np.var(ti[ti >= 0]):
                        tn = -tn
                        ti = -ti
                        pi = -pi

                if not shush:
                    print(f"# Iterations for PC #{a + 1}: {num_it}")

                if a == 0:
                    T = tn.reshape(-1, 1)
                    P = pi
                else:
                    T = np.hstack((T, tn.reshape(-1, 1)))
                    P = np.hstack((P, pi))

                # Deflate X
                X_ = (X_ - ti @ pi.T) * not_Xmiss

                if a == 0:
                    r2 = 1 - np.sum(X_**2) / TSS
                    r2pv = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
                else:
                    r2 = np.hstack((r2, 1 - np.sum(X_**2) / TSS))
                    aux_ = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
                    r2pv = np.hstack((r2pv, aux_))
            else:
                num_it += 1
                ti = tn.reshape(-1, 1)

        if a == 0:
            numIT = num_it
        else:
            numIT = np.hstack((numIT, num_it))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2[a] = r2[a] - r2[a - 1]
        r2pv[:, a] = r2pv[:, a] - r2pv[:, a - 1]

    pca_obj = {"T": T, "P": P, "r2x": r2, "r2xpv": r2pv, "mx": x_mean, "sx": x_std}
    if not isinstance(obsidX, bool):
        pca_obj["obsidX"] = obsidX
        pca_obj["varidX"] = varidX

    # Add diagnostics
    _add_pca_diagnostics(pca_obj, T, X_, A, shush)

    return pca_obj


def _pca_nlp(
    X: np.ndarray | pd.DataFrame,
    X_: np.ndarray,
    A: int,
    mcs: bool | str,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    not_Xmiss: np.ndarray,
    obsidX: list | bool,
    varidX: list | bool,
    shush: bool,
) -> dict:
    """PCA using NLP optimization for missing data.

    Uses Pyomo/IPOPT for optimization per Lopez-Negrete et al.
    J. Chemometrics 2010; 24: 301-311.
    """
    from shutil import which

    if not shush:
        print(f"phi.pca using NLP with Ipopt executed on: {datetime.datetime.now()}")

    # Get initial solution from NIPALS
    pcaobj_ = pca_(X, A, mcs=mcs, md_algorithm="nipals", shush=True)
    pcaobj_ = prep_pca_4_MDbyNLP(pcaobj_, X_)

    TSS = np.sum(X_**2)
    TSSpv = np.sum(X_**2, axis=0)

    # Set up Pyomo model
    model = ConcreteModel()
    model.A = Set(initialize=pcaobj_["pyo_A"])
    model.N = Set(initialize=pcaobj_["pyo_N"])
    model.O = Set(initialize=pcaobj_["pyo_O"])
    model.P = Var(model.N, model.A, within=Reals, initialize=pcaobj_["pyo_P_init"])
    model.T = Var(model.O, model.A, within=Reals, initialize=pcaobj_["pyo_T_init"])
    model.psi = Param(model.O, model.N, initialize=pcaobj_["pyo_psi"])
    model.X = Param(model.O, model.N, initialize=pcaobj_["pyo_X"])
    model.delta = Param(
        model.A, model.A, initialize=lambda model, a1, a2: 1.0 if a1 == a2 else 0
    )

    # Constraint 20b: P'P = I
    def _c20b_con(model, a1, a2):
        return sum(model.P[j, a1] * model.P[j, a2] for j in model.N) == model.delta[a1, a2]

    model.c20b = Constraint(model.A, model.A, rule=_c20b_con)

    # Constraint 20c: T'T diagonal (orthogonal scores)
    def _20c_con(model, a1, a2):
        if a2 < a1:
            return sum(model.T[o, a1] * model.T[o, a2] for o in model.O) == 0
        else:
            return Constraint.Skip

    model.c20c = Constraint(model.A, model.A, rule=_20c_con)

    # Constraint 20d: Mean-centered scores
    def mean_zero(model, i):
        return sum(model.T[o, i] for o in model.O) == 0

    model.eq3 = Constraint(model.A, rule=mean_zero)

    # Objective: Minimize reconstruction error
    def _eq_20a_obj(model):
        return sum(
            sum(
                (
                    model.X[o, n]
                    - model.psi[o, n] * sum(model.T[o, a] * model.P[n, a] for a in model.A)
                )
                ** 2
                for n in model.N
            )
            for o in model.O
        )

    model.obj = Objective(rule=_eq_20a_obj)

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

    # Extract results
    T = np.array([[value(model.T[o, a]) for a in model.A] for o in model.O])
    P = np.array([[value(model.P[n, a]) for a in model.A] for n in model.N])

    # Calculate R²
    for a in range(A):
        ti = T[:, [a]]
        pi = P[:, [a]]
        # Rotate if needed
        if np.var(ti[ti < 0]) > np.var(ti[ti >= 0]):
            ti = -ti
            pi = -pi
            T[:, [a]] = -T[:, [a]]
            P[:, [a]] = -P[:, [a]]

        X_ = (X_ - ti @ pi.T) * not_Xmiss

        if a == 0:
            r2 = 1 - np.sum(X_**2) / TSS
            r2pv = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
        else:
            r2 = np.hstack((r2, 1 - np.sum(X_**2) / TSS))
            aux_ = (1 - np.sum(X_**2, axis=0) / TSSpv).reshape(-1, 1)
            r2pv = np.hstack((r2pv, aux_))

    # Convert cumulative to per-component
    for a in range(A - 1, 0, -1):
        r2[a] = r2[a] - r2[a - 1]
        r2pv[:, a] = r2pv[:, a] - r2pv[:, a - 1]

    pca_obj = {"T": T, "P": P, "r2x": r2, "r2xpv": r2pv, "mx": x_mean, "sx": x_std}
    if not isinstance(obsidX, bool):
        pca_obj["obsidX"] = obsidX
        pca_obj["varidX"] = varidX

    # Add diagnostics
    _add_pca_diagnostics(pca_obj, T, X_, A, shush)

    return pca_obj


def _add_pca_diagnostics(
    pca_obj: dict, T: np.ndarray, X_residual: np.ndarray, A: int, shush: bool
) -> None:
    """Add diagnostic statistics to PCA object."""
    eigs = np.var(T, axis=0)
    r2 = pca_obj["r2x"]
    r2xc = np.cumsum(r2)

    if not shush:
        print("-" * 62)
        print("PC #      Eig        R2X       sum(R2X) ")
        if A > 1:
            for a in range(A):
                print(f"PC #{a + 1}:   {eigs[a]:8.3f}    {r2[a]:.3f}     {r2xc[a]:.3f}")
        else:
            print(f"PC #1:   {eigs[0]:8.3f}    {r2:.3f}     {r2xc[0]:.3f}")
        print("-" * 62)

    # Hotelling's T²
    T2 = hott2(pca_obj, Tnew=T)
    n = T.shape[0]
    T2_lim99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, n - A)
    T2_lim95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, n - A)

    # SPE
    speX = np.sum(X_residual**2, axis=1, keepdims=True)
    speX_lim95, speX_lim99 = spe_ci(speX)

    pca_obj["T2"] = T2
    pca_obj["T2_lim99"] = T2_lim99
    pca_obj["T2_lim95"] = T2_lim95
    pca_obj["speX"] = speX
    pca_obj["speX_lim99"] = speX_lim99
    pca_obj["speX_lim95"] = speX_lim95


def pca_pred(
    Xnew: np.ndarray | pd.DataFrame,
    pcaobj: dict,
    *,
    algorithm: str = "p2mp",
) -> dict:
    """Evaluate new data using an already built PCA model.

    Parameters
    ----------
    Xnew : np.ndarray or pd.DataFrame
        New data to project onto the PCA model.
    pcaobj : dict
        PCA model object created by pca().
    algorithm : str, default="p2mp"
        Algorithm for handling missing data:
        - "p2mp": Projection to Model Plane method

    Returns
    -------
    dict
        Prediction results containing:
        - Xhat: Reconstructed X values
        - Tnew: Projected scores
        - speX: SPE for new observations
        - T2: Hotelling's T² for new observations
    """
    if isinstance(Xnew, np.ndarray):
        X_ = Xnew.copy()
        if X_.ndim == 1:
            X_ = np.reshape(X_, (1, -1))
    elif isinstance(Xnew, pd.DataFrame):
        X_ = np.array(Xnew.values[:, 1:]).astype(float)

    X_nan_map = np.isnan(X_)

    if not X_nan_map.any():
        # Complete data - direct projection
        X_mcs = X_ - np.tile(pcaobj["mx"], (X_.shape[0], 1))
        X_mcs = X_mcs / np.tile(pcaobj["sx"], (X_.shape[0], 1))
        tnew = X_mcs @ pcaobj["P"]
        xhat = (tnew @ pcaobj["P"].T) * np.tile(pcaobj["sx"], (X_.shape[0], 1)) + np.tile(
            pcaobj["mx"], (X_.shape[0], 1)
        )
        var_t = (pcaobj["T"].T @ pcaobj["T"]) / pcaobj["T"].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        spe = X_mcs - (tnew @ pcaobj["P"].T)
        spe = np.sum(spe**2, axis=1, keepdims=True)
        xpred = {"Xhat": xhat, "Tnew": tnew, "speX": spe, "T2": htt2}

    elif algorithm == "p2mp":
        # Projection to Model Plane for missing data
        not_Xmiss = (~X_nan_map).astype(int)

        Xmcs = (X_ - np.tile(pcaobj["mx"], (X_.shape[0], 1))) / np.tile(
            pcaobj["sx"], (X_.shape[0], 1)
        )
        Xmcs, dummy = n2z(Xmcs.copy())

        for i in range(Xmcs.shape[0]):
            row_missing_map = not_Xmiss[[i], :]
            tempP = pcaobj["P"] * np.tile(row_missing_map.T, (1, pcaobj["P"].shape[1]))
            PTP = tempP.T @ tempP
            try:
                tnew_, resid, rank, s = np.linalg.lstsq(
                    PTP, (tempP.T @ Xmcs[[i], :].T), rcond=None
                )
            except Exception:
                tnew_ = np.linalg.pinv(PTP) @ tempP.T @ Xmcs[[i], :].T

            if i == 0:
                tnew = tnew_.T
            else:
                tnew = np.vstack((tnew, tnew_.T))

        xhat = (tnew @ pcaobj["P"].T) * np.tile(pcaobj["sx"], (X_.shape[0], 1)) + np.tile(
            pcaobj["mx"], (X_.shape[0], 1)
        )
        var_t = (pcaobj["T"].T @ pcaobj["T"]) / pcaobj["T"].shape[0]
        htt2 = np.sum((tnew @ np.linalg.inv(var_t)) * tnew, axis=1)
        spe = Xmcs - (tnew @ pcaobj["P"].T)
        spe = spe * not_Xmiss
        spe = np.sum(spe**2, axis=1, keepdims=True)
        xpred = {"Xhat": xhat, "Tnew": tnew, "speX": spe, "T2": htt2}

    return xpred

