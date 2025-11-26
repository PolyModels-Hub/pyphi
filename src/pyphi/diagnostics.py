"""
Diagnostics and Analysis Tools for PyPhi.

This module provides diagnostic tools for analyzing PCA/PLS models:
- Contribution analysis (T², SPE, scores)
- Varimax rotation for interpretable loadings
- Bootstrap PLS for uncertainty quantification
- Polynomial model building with PLS-assisted variable selection

Author: Salvador Garcia-Munoz (sgarciam@ic.ac.uk, salvadorgarciamunoz@gmail.com)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy import eye, asarray, dot, sum as np_sum, diag
from numpy.linalg import svd
from scipy.stats import norm
from scipy.optimize import fsolve

from .utils import n2z
from ._internal import _Ab_btbinv


# =============================================================================
# Contribution Analysis
# =============================================================================


def contributions(
    mvmobj: dict,
    X: np.ndarray | pd.DataFrame,
    cont_type: str,
    *,
    Y: np.ndarray | pd.DataFrame | bool = False,
    from_obs: int | list[int] | bool = False,
    to_obs: int | list[int] | bool = False,
    lv_space: int | list[int] | bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Calculate contributions to diagnostics.

    Computes variable contributions to Hotelling's T², SPE, or scores
    for specified observations.

    Parameters
    ----------
    mvmobj : dict
        A model dictionary created by pca() or pls().
    X : np.ndarray or pd.DataFrame
        X data matrix. If DataFrame, first column is observation ID.
    cont_type : str
        Type of contribution to calculate:
        - 'ht2': Contributions to Hotelling's T²
        - 'spe': Contributions to SPE space
        - 'scores': Contributions to scores
    Y : np.ndarray or pd.DataFrame or False, default=False
        Y data matrix (optional, for SPE contributions in PLS models).
    from_obs : int, list of int, or False, default=False
        Observation(s) to offset from. If False, contributions are
        calculated with respect to the mean.
    to_obs : int, list of int, or False, default=False
        Observation(s) to calculate contributions for.
        Note: from_obs is ignored when cont_type='spe'.
    lv_space : int, list of int, or False, default=False
        Latent spaces over which to do the calculations.
        Only applicable to 'ht2' and 'scores'. If False, all
        dimensions are considered.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        For 'ht2' and 'scores': array of contributions (1 x n_variables).
        For 'spe': contributions to X (and Y if provided).

    Examples
    --------
    >>> model = pca(X_train, 3)
    >>> contrib = contributions(model, X_test, 'ht2', to_obs=[0, 1, 2])
    >>> contrib_x, contrib_y = contributions(model, X_test, 'spe', Y=Y_test, to_obs=[5])
    """
    # Import prediction functions here to avoid circular imports
    from .pca import pca_pred
    from .pls import pls_pred

    # Handle lv_space parameter
    if isinstance(lv_space, bool):
        lv_space = list(range(mvmobj["T"].shape[1]))
    elif isinstance(lv_space, int):
        lv_space = (np.array([lv_space]) - 1).tolist()
    elif isinstance(lv_space, list):
        lv_space = (np.array(lv_space) - 1).tolist()

    # Handle to_obs parameter
    if isinstance(to_obs, int):
        to_obs = np.array([to_obs]).tolist()
    elif isinstance(to_obs, list):
        to_obs = np.array(to_obs).tolist()

    # Handle from_obs parameter
    if not isinstance(from_obs, bool):
        if isinstance(from_obs, int):
            from_obs = np.array([from_obs]).tolist()
        elif isinstance(from_obs, list):
            from_obs = np.array(from_obs).tolist()

    # Extract X data
    if isinstance(X, np.ndarray):
        X_ = X.copy()
    elif isinstance(X, pd.DataFrame):
        X_ = np.array(X.values[:, 1:]).astype(float)

    # Extract Y data if provided
    if not isinstance(Y, bool):
        if isinstance(Y, np.ndarray):
            Y_ = Y.copy()
        elif isinstance(Y, pd.DataFrame):
            Y_ = np.array(Y.values[:, 1:]).astype(float)

    if cont_type == "ht2" or cont_type == "scores":
        # Scale X data
        X_ = (X_ - np.tile(mvmobj["mx"], (X_.shape[0], 1))) / (
            np.tile(mvmobj["sx"], (X_.shape[0], 1))
        )
        X_, dummy = n2z(X_)
        t_stdevs = np.std(mvmobj["T"], axis=0, ddof=1)

        # Get appropriate loadings
        if "Q" in mvmobj:
            loadings = mvmobj["Ws"]
        else:
            loadings = mvmobj["P"]

        to_obs_mean = np.mean(X_[to_obs, :], axis=0, keepdims=True)
        to_cont = np.zeros((1, X_.shape[1]))

        for a in lv_space:
            aux_ = (to_obs_mean * np.abs(loadings[:, a].T)) / t_stdevs[a]
            if cont_type == "scores":
                to_cont = to_cont + aux_
            else:
                to_cont = to_cont + aux_**2

        if not isinstance(from_obs, bool):
            from_obs_mean = np.mean(X_[from_obs, :], axis=0, keepdims=True)
            from_cont = np.zeros((1, X_.shape[1]))
            for a in lv_space:
                aux_ = (from_obs_mean * np.abs(loadings[:, a].T)) / t_stdevs[a]
                if cont_type == "scores":
                    from_cont = from_cont + aux_
                else:
                    from_cont = from_cont + aux_**2
            calc_contribution = to_cont - from_cont
            return calc_contribution
        else:
            return to_cont

    elif cont_type == "spe":
        X_ = X_[to_obs, :]
        if "Q" in mvmobj:
            pred = pls_pred(X_, mvmobj)
        else:
            pred = pca_pred(X_, mvmobj)
        Xhat = pred["Xhat"]
        Xhatmcs = (Xhat - np.tile(mvmobj["mx"], (Xhat.shape[0], 1))) / (
            np.tile(mvmobj["sx"], (Xhat.shape[0], 1))
        )
        X_ = (X_ - np.tile(mvmobj["mx"], (X_.shape[0], 1))) / (
            np.tile(mvmobj["sx"], (X_.shape[0], 1))
        )
        Xerror = X_ - Xhatmcs
        Xerror, dummy = n2z(Xerror)
        contsX = ((Xerror) ** 2) * np.sign(Xerror)
        contsX = np.mean(contsX, axis=0, keepdims=True)

        if not isinstance(Y, bool):
            Y_ = Y_[to_obs, :]
            Yhat = pred["Yhat"]
            Yhatmcs = (Yhat - np.tile(mvmobj["my"], (Yhat.shape[0], 1))) / (
                np.tile(mvmobj["sy"], (Yhat.shape[0], 1))
            )
            Y_ = (Y_ - np.tile(mvmobj["my"], (Y_.shape[0], 1))) / (
                np.tile(mvmobj["sy"], (Y_.shape[0], 1))
            )
            Yerror = Y_ - Yhatmcs
            Yerror, dummy = n2z(Yerror)
            contsY = ((Yerror) ** 2) * np.sign(Yerror)
            contsY = np.mean(contsY, axis=0, keepdims=True)
            return contsX, contsY
        else:
            return contsX

    else:
        raise ValueError(f"Unknown cont_type: {cont_type}. Use 'ht2', 'spe', or 'scores'.")


# =============================================================================
# Varimax Rotation
# =============================================================================


def varimax_(
    X: np.ndarray,
    gamma: float = 1.0,
    q: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """Internal varimax rotation algorithm.

    Performs varimax rotation on a loading matrix to achieve
    simple structure (maximize variance of squared loadings).

    Parameters
    ----------
    X : np.ndarray
        Loading matrix of shape (p, k) to rotate.
    gamma : float, default=1.0
        Rotation parameter. gamma=1.0 gives standard varimax.
    q : int, default=20
        Maximum number of iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    np.ndarray
        Rotated loading matrix of shape (p, k).

    Notes
    -----
    This is an internal function. Use varimax_rotation() for
    rotating complete model objects.
    """
    p, k = X.shape
    R = eye(k)
    d = 0

    for i in range(q):
        d_ = d
        Lambda = dot(X, R)
        u, s, vh = svd(
            dot(
                X.T,
                asarray(Lambda) ** 3
                - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda)))),
            )
        )
        R = dot(u, vh)
        d = np_sum(s)
        if d_ != 0 and d / d_ < 1 + tol:
            break

    return dot(X, R)


def varimax_rotation(
    mvm_obj: dict,
    X: np.ndarray | pd.DataFrame,
    *,
    Y: np.ndarray | pd.DataFrame | bool = False,
) -> dict:
    """Apply varimax rotation to a PCA or PLS model.

    Performs varimax rotation on loadings and recalculates scores
    and R² values. This improves interpretability by achieving
    simple structure in the loadings.

    Parameters
    ----------
    mvm_obj : dict
        A PCA or PLS model object created by pca() or pls().
    X : np.ndarray or pd.DataFrame
        X data matrix used to build the model.
        If DataFrame, first column is observation ID.
    Y : np.ndarray or pd.DataFrame or False, default=False
        Y data matrix (required for PLS models).
        If DataFrame, first column is observation ID.

    Returns
    -------
    dict
        The rotated model with updated:
        - 'P': Rotated P loadings
        - 'T': Rotated T scores
        - 'r2x': Recalculated R² for X
        - 'r2xpv': Recalculated per-variable R² for X
        For PLS models, also:
        - 'W': Rotated W weights
        - 'Q': Rotated Q loadings
        - 'U': Rotated U scores
        - 'Ws': Rotated Ws (W-star) weights
        - 'r2y': Recalculated R² for Y
        - 'r2ypv': Recalculated per-variable R² for Y

    Examples
    --------
    >>> model = pca(X_train, 3)
    >>> rotated_model = varimax_rotation(model, X_train)

    >>> model = pls(X_train, Y_train, 3)
    >>> rotated_model = varimax_rotation(model, X_train, Y=Y_train)
    """
    mvmobj = mvm_obj.copy()

    # Extract X data
    if isinstance(X, np.ndarray):
        X_ = X.copy()
    if isinstance(X, pd.DataFrame):
        X_ = X.values[:, 1:].astype(float)

    # Extract Y data if provided
    if isinstance(Y, np.ndarray):
        Y_ = Y.copy()
    if isinstance(Y, pd.DataFrame):
        Y_ = Y.values[:, 1:].astype(float)

    # Scale X
    X_ = (X_ - np.tile(mvmobj["mx"], (X_.shape[0], 1))) / np.tile(
        mvmobj["sx"], (X_.shape[0], 1)
    )
    not_Xmiss = ~(np.isnan(X_)) * 1
    X_, Xmap = n2z(X_)
    TSSX = np.sum(X_**2)
    TSSXpv = np.sum(X_**2, axis=0)

    if not isinstance(Y, bool):
        Y_ = (Y_ - np.tile(mvmobj["my"], (Y_.shape[0], 1))) / np.tile(
            mvmobj["sy"], (Y_.shape[0], 1)
        )
        not_Ymiss = ~(np.isnan(Y_)) * 1
        Y_, Ymap = n2z(Y_)
        TSSY = np.sum(Y_**2)
        TSSYpv = np.sum(Y_**2, axis=0)

    A = mvmobj["T"].shape[1]

    if "Q" in mvmobj:
        # PLS model
        Wrot = varimax_(mvmobj["W"])
        Trot = []
        Prot = []
        Qrot = []
        Urot = []

        for a in np.arange(A):
            ti = _Ab_btbinv(X_, Wrot[:, a], not_Xmiss)
            pi = _Ab_btbinv(X_.T, ti, not_Xmiss.T)
            qi = _Ab_btbinv(Y_.T, ti, not_Ymiss.T)
            ui = _Ab_btbinv(Y_, qi, not_Ymiss)
            X_ = (X_ - ti @ pi.T) * not_Xmiss
            Y_ = (Y_ - ti @ qi.T) * not_Ymiss

            if a == 0:
                r2Xpv = np.zeros((1, len(TSSXpv))).reshape(-1)
                r2X = 1 - np.sum(X_**2) / TSSX
                r2Xpv[TSSXpv > 0] = 1 - (
                    np.sum(X_**2, axis=0)[TSSXpv > 0] / TSSXpv[TSSXpv > 0]
                )
                r2Xpv = r2Xpv.reshape(-1, 1)

                r2Ypv = np.zeros((1, len(TSSYpv))).reshape(-1)
                r2Y = 1 - np.sum(Y_**2) / TSSY
                r2Ypv[TSSYpv > 0] = 1 - (
                    np.sum(Y_**2, axis=0)[TSSYpv > 0] / TSSYpv[TSSYpv > 0]
                )
                r2Ypv = r2Ypv.reshape(-1, 1)
            else:
                r2X = np.hstack((r2X, 1 - np.sum(X_**2) / TSSX))
                aux_ = np.zeros((1, len(TSSXpv))).reshape(-1)
                aux_[TSSXpv > 0] = 1 - (
                    np.sum(X_**2, axis=0)[TSSXpv > 0] / TSSXpv[TSSXpv > 0]
                )
                aux_ = aux_.reshape(-1, 1)
                r2Xpv = np.hstack((r2Xpv, aux_))

                r2Y = np.hstack((r2Y, 1 - np.sum(Y_**2) / TSSY))
                aux_ = np.zeros((1, len(TSSYpv))).reshape(-1)
                aux_[TSSYpv > 0] = 1 - (
                    np.sum(Y_**2, axis=0)[TSSYpv > 0] / TSSYpv[TSSYpv > 0]
                )
                aux_ = aux_.reshape(-1, 1)
                r2Ypv = np.hstack((r2Ypv, aux_))

            Trot.append(ti)
            Prot.append(pi)
            Qrot.append(qi)
            Urot.append(ui)

        # Convert cumulative R² to per-component R²
        for a in list(range(A - 1, 0, -1)):
            r2X[a] = r2X[a] - r2X[a - 1]
            r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]
            r2Y[a] = r2Y[a] - r2Y[a - 1]
            r2Ypv[:, a] = r2Ypv[:, a] - r2Ypv[:, a - 1]

        Trot = np.array(Trot).T
        Prot = np.array(Prot).T
        Qrot = np.array(Qrot).T
        Urot = np.array(Urot).T
        Trot = Trot[0]
        Prot = Prot[0]
        Qrot = Qrot[0]
        Urot = Urot[0]
        Wsrot = Wrot @ np.linalg.pinv(Prot.T @ Wrot)

        mvmobj["W"] = Wrot
        mvmobj["T"] = Trot
        mvmobj["P"] = Prot
        mvmobj["Q"] = Qrot
        mvmobj["U"] = Urot
        mvmobj["Ws"] = Wsrot
        mvmobj["r2x"] = r2X
        mvmobj["r2xpv"] = r2Xpv
        mvmobj["r2y"] = r2Y
        mvmobj["r2ypv"] = r2Ypv

    else:
        # PCA model
        Prot = varimax_(mvmobj["P"])
        Trot = []

        for a in np.arange(A):
            ti = _Ab_btbinv(X_, Prot[:, a], not_Xmiss)
            Trot.append(ti)
            pi = Prot[:, [a]]
            X_ = (X_ - ti @ pi.T) * not_Xmiss

            if a == 0:
                r2Xpv = np.zeros((1, len(TSSXpv))).reshape(-1)
                r2X = 1 - np.sum(X_**2) / TSSX
                r2Xpv[TSSXpv > 0] = 1 - (
                    np.sum(X_**2, axis=0)[TSSXpv > 0] / TSSXpv[TSSXpv > 0]
                )
                r2Xpv = r2Xpv.reshape(-1, 1)
            else:
                r2X = np.hstack((r2X, 1 - np.sum(X_**2) / TSSX))
                aux_ = np.zeros((1, len(TSSXpv))).reshape(-1)
                aux_[TSSXpv > 0] = 1 - (
                    np.sum(X_**2, axis=0)[TSSXpv > 0] / TSSXpv[TSSXpv > 0]
                )
                aux_ = aux_.reshape(-1, 1)
                r2Xpv = np.hstack((r2Xpv, aux_))

        Trot = np.array(Trot).T

        # Convert cumulative R² to per-component R²
        for a in list(range(A - 1, 0, -1)):
            r2X[a] = r2X[a] - r2X[a - 1]
            r2Xpv[:, a] = r2Xpv[:, a] - r2Xpv[:, a - 1]

        mvmobj["P"] = Prot
        mvmobj["T"] = Trot[0]
        mvmobj["r2x"] = r2X
        mvmobj["r2xpv"] = r2Xpv

    return mvmobj


# =============================================================================
# Bootstrap PLS
# =============================================================================


def bootstrap_pls(
    X: np.ndarray | pd.DataFrame,
    Y: np.ndarray | pd.DataFrame,
    num_latents: int,
    num_samples: int,
    **kwargs: Any,
) -> list[dict]:
    """Generate bootstrap PLS models for uncertainty quantification.

    Creates multiple PLS models by resampling the training data with
    replacement. These can be used to estimate prediction uncertainty.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        X data matrix. If DataFrame, first column is observation ID.
    Y : np.ndarray or pd.DataFrame
        Y data matrix. If DataFrame, first column is observation ID.
    num_latents : int
        Number of latent variables for each PLS model.
    num_samples : int
        Number of bootstrap samples to generate.
    **kwargs
        Additional arguments passed to pls() (e.g., mcsX, mcsY).

    Returns
    -------
    list of dict
        List of PLS model dictionaries from bootstrap samples.

    Examples
    --------
    >>> boot_models = bootstrap_pls(X_train, Y_train, num_latents=3, num_samples=100)
    >>> predictions = bootstrap_pls_pred(X_new, boot_models)
    """
    from .pls import pls

    # Extract numpy arrays
    if isinstance(X, pd.DataFrame):
        Dm_values = X.values[:, 1:].astype(float)
    else:
        Dm_values = X

    if isinstance(Y, pd.DataFrame):
        Y_values = Y.values[:, 1:].astype(float)
    else:
        Y_values = Y

    boot_pls_obj = []
    for _ in range(num_samples):
        sample_indexes = np.random.randint(Dm_values.shape[0], None, Dm_values.shape[0])
        boot_X = Dm_values[sample_indexes]
        boot_Y = Y_values[sample_indexes]
        boot_pls_obj.append(pls(boot_X, boot_Y, num_latents, shush=True, **kwargs))

    return boot_pls_obj


def bootstrap_pls_pred(
    X_new: np.ndarray | pd.DataFrame,
    bootstrap_pls_obj: list[dict],
    quantiles: list[float] = [0.025, 0.975],
) -> list[np.ndarray]:
    """Make predictions with quantiles using bootstrapped PLS models.

    Uses the ensemble of bootstrap PLS models to generate prediction
    intervals assuming Gaussian errors.

    Parameters
    ----------
    X_new : np.ndarray or pd.DataFrame
        New X data for prediction. If DataFrame, first column is observation ID.
    bootstrap_pls_obj : list of dict
        List of bootstrap PLS models from bootstrap_pls().
    quantiles : list of float, default=[0.025, 0.975]
        Quantiles to compute (values between 0 and 1).
        Default gives 95% prediction interval.

    Returns
    -------
    list of np.ndarray
        List of predicted values at each specified quantile.

    Raises
    ------
    ValueError
        If any quantile is not between 0 and 1.

    Examples
    --------
    >>> boot_models = bootstrap_pls(X_train, Y_train, 3, 100)
    >>> lower, upper = bootstrap_pls_pred(X_new, boot_models, [0.025, 0.975])
    """
    from .pls import pls_pred

    for quantile in quantiles:
        if quantile >= 1 or quantile <= 0:
            raise ValueError("Quantiles must be between zero and one")

    means = []
    sds = []
    for pls_obj in bootstrap_pls_obj:
        means.append(pls_pred(X_new, pls_obj)["Yhat"])
        sds.append(np.sqrt(pls_obj["speY"].mean()))

    means = np.array(means).squeeze()
    sds = np.array(sds)
    dist = norm(means, sds[:, None])

    ppf = []
    for quantile in quantiles:

        def cdf(x):
            return dist.cdf(x).mean(axis=0) - np.ones_like(x) * quantile

        ppf.append(fsolve(cdf, means.mean(axis=0)))

    return ppf


# =============================================================================
# Polynomial Model Building (Internal Helpers)
# =============================================================================


def _findstr(string: str) -> list[int]:
    """Find positions of * and / operators in a string.

    Parameters
    ----------
    string : str
        Expression string to parse.

    Returns
    -------
    list of int
        Indices of operator positions.
    """
    indx = []
    for i, s in enumerate(string):
        if s == "*" or s == "/":
            indx.append(i)
    return indx


def _evalvar(data: pd.DataFrame, vname: str) -> np.ndarray | bool:
    """Evaluate a variable expression from a DataFrame.

    Supports power notation (e.g., 'var^2') and simple variable names.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the variables.
    vname : str
        Variable name, optionally with power (e.g., 'Temperature^2').

    Returns
    -------
    np.ndarray or False
        Column values (optionally raised to power), or False if variable not found.
    """
    if vname.find("^") > 0:
        actual_vname = vname[: vname.find("^")]
        actual_vname = actual_vname.strip()
        if actual_vname in data.columns[1:]:
            power = float(vname[vname.find("^") + 1 :])
            vals = data[actual_vname].values.reshape(-1, 1) ** power
        else:
            vals = False
    else:
        actual_vname = vname.strip()
        if actual_vname in data.columns[1:]:
            vals = data[actual_vname].values.reshape(-1, 1)
        else:
            vals = False
    return vals


def _writeeq(beta_: np.ndarray, features_: list[str]) -> str:
    """Format regression equation as a string.

    Parameters
    ----------
    beta_ : np.ndarray
        Regression coefficients.
    features_ : list of str
        Feature names (including 'Bias' for intercept).

    Returns
    -------
    str
        Formatted equation string.
    """
    eq_str = []
    for b, f, i in zip(beta_, features_, np.arange(len(features_))):
        if f == "Bias":
            if b < 0:
                eq_str.extend(str(b))
            else:
                eq_str.extend(" + " + str(b))
        else:
            if b < 0 or i == 0:
                eq_str.extend(str(b) + " * " + f)
            else:
                eq_str.extend(" + " + str(b) + " * " + f)
    return "".join(eq_str)


# =============================================================================
# Polynomial Model Building
# =============================================================================


def build_polynomial(
    data: pd.DataFrame,
    factors: list[str],
    response: str,
    *,
    bias_term: bool = True,
    show_plots: bool = True,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, str]:
    """Build polynomial regression with PLS-assisted variable selection.

    Creates a linear regression model with variable selection guided by
    VIP (Variable Importance in Projection) from PLS. Supports polynomial
    terms, interactions, and ratios.

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame with first column as observation ID.
    factors : list of str
        List of factors to include in the expression. Supports:
        - Simple variables: 'Variable 1'
        - Powers: 'Variable 1^2'
        - Interactions: 'Variable 1 * Variable 2'
        - Ratios: 'Variable 1 / Variable 2'
        - Combined: 'Variable 1^2 / Variable 2'
    response : str
        Response variable name (must be a column in data).
    bias_term : bool, default=True
        Whether to include an intercept term.
    show_plots : bool, default=True
        Whether to display VIP and RMSE plots.

    Returns
    -------
    tuple
        - betasOLSlssq: Regression coefficients
        - factors_out: Factor names (including 'Bias' if bias_term=True)
        - Xaug: Augmented X matrix
        - Y: Response values
        - eqstr: Formatted equation string

    Examples
    --------
    >>> factors = ['Temperature', 'Temperature^2', 'Pressure', 'Temperature * Pressure']
    >>> betas, names, X, Y, eq = build_polynomial(data, factors, 'Yield')
    >>> print(eq)
    '0.5 * Temperature + -0.1 * Temperature^2 + 0.3 * Pressure + ...'
    """
    import matplotlib.pyplot as plt
    from .pls import pls, pls_pred

    for j, f in enumerate(factors):
        # Search for * or / operators
        if f.find("*") > 0 or f.find("/") > 0:
            # Find all locations for * or /
            indx = _findstr(f)
            for ii, i in enumerate(indx):
                if ii == 0:
                    vname1 = f[0:i]
                    if len(indx) > 1:
                        vname2 = f[i + 1 : indx[1]]
                    else:
                        vname2 = f[i + 1 :]

                    vals1 = _evalvar(data, vname1)
                    vals2 = _evalvar(data, vname2)
                    if f[i] == "*":
                        xcol = vals1 * vals2
                    elif f[i] == "/":
                        xcol = vals1 / vals2
                else:
                    if len(indx) == ii + 1:
                        vname = f[i + 1 :]
                    else:
                        vname = f[i + 1 : indx[ii + 1]]
                    vals = _evalvar(data, vname)
                    if f[i] == "*":
                        xcol = xcol * vals
                    elif f[i] == "/":
                        xcol = xcol / vals
            if j == 0:
                X = xcol
            else:
                X = np.hstack((X, xcol))
        else:
            if j == 0:
                X = _evalvar(data, f)
            else:
                temp = _evalvar(data, f)
                X = np.hstack((X, temp))

    print("Built X from factors")
    X_df = pd.DataFrame(X, columns=factors)
    X_df.insert(0, data.columns[0], data[data.columns[0]].values)
    Y_df = data[[data.columns[0], response]]
    Y = data[response].values

    pls_obj = pls(X_df, Y_df, len(factors), shush=True)
    Ypred = pls_pred(X_df, pls_obj)
    Ypred = Ypred["Yhat"]
    RMSE = [np.sqrt(np.mean((Y_df.values[:, 1:].astype(float) - Ypred) ** 2))]

    vip = np.sum(
        np.abs(pls_obj["Ws"] * np.tile(pls_obj["r2y"], (pls_obj["Ws"].shape[0], 1))),
        axis=1,
    )
    vip = np.reshape(vip, -1)
    sort_indx = np.argsort(-vip, axis=0)
    sort_asc_indx = np.argsort(vip, axis=0)
    vip = vip[sort_indx]

    sorted_factors = []
    for i in sort_indx:
        sorted_factors.append(factors[i])

    if show_plots:
        plt.figure()
        plt.bar(np.arange(len(sorted_factors)), vip)
        plt.xticks(np.arange(len(sorted_factors)), labels=sorted_factors, rotation=60)
        plt.ylabel("VIP")
        plt.xlabel("Factors")
        plt.tight_layout()

    sorted_asc_factors = []
    for i in sort_asc_indx:
        sorted_asc_factors.append(factors[i])

    X_df_m = X_df.copy()
    for f in sorted_asc_factors:
        if f != sorted_asc_factors[-1]:
            X_df_m.drop(f, axis=1, inplace=True)
            pls_obj = pls(X_df_m, Y_df, X_df_m.shape[1] - 1, shush=True)
            Ypred = pls_pred(X_df_m, pls_obj)
            Ypred = Ypred["Yhat"]
            RMSE.append(
                np.sqrt(np.mean((Y_df.values[:, 1:].astype(float) - Ypred) ** 2))
            )

    sorted_asc_factors_lbl = ["Full"]
    for i in sort_asc_indx[:-1]:
        sorted_asc_factors_lbl.append(factors[i])

    if show_plots:
        plt.figure()
        plt.bar(np.arange(len(factors)), RMSE)
        plt.xticks(np.arange(len(factors)), labels=sorted_asc_factors_lbl, rotation=60)
        plt.ylabel("RMSE (" + response + ")")
        plt.xlabel("Factors removed from model")
        plt.tight_layout()

    if bias_term:
        Xaug = np.hstack((X, np.ones((X.shape[0], 1))))
        factors_out = factors.copy()
        factors_out.append("Bias")
    else:
        Xaug = X
        factors_out = factors.copy()

    betasOLSlssq, r1, r2, r3 = np.linalg.lstsq(Xaug, Y, rcond=None)
    eqstr = _writeeq(betasOLSlssq, factors_out)

    return betasOLSlssq, factors_out, Xaug, Y, eqstr


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    "contributions",
    "varimax_",
    "varimax_rotation",
    "bootstrap_pls",
    "bootstrap_pls_pred",
    "build_polynomial",
]

