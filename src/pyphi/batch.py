# -*- coding: utf-8 -*-
"""
Batch analysis utilities for PyPhi.

This module is the refactored home for the legacy top-level `pyphi_batch.py`.
The legacy file is intentionally kept in the repo for reference, but this
module is what should be imported and used going forward:

    from pyphi import batch as phibatch

The implementation aims to preserve the original behavior while using the
refactored `pyphi` package internals (no `import pyphi as phi` here to avoid
circular imports).
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .advanced_pls import mbpls
from .diagnostics import contributions as _phi_contributions
from .pca import pca, pca_pred
from .pls import pls, pls_pred
from .utils import (
    clean_empty_rows as _clean_empty_rows_matrix,
    clean_low_variances,
    f95,
    f99,
    single_score_conf_int,
    spe_ci,
    unique as _unique_df,
)


# Sequence of color blind friendly colors.
cb_color_seq = ["b", "r", "m", "navy", "bisque", "silver", "aqua", "pink", "gray"]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=cb_color_seq)


def _has_phase_col(bdata: pd.DataFrame) -> bool:
    return (bdata.columns[1] == "PHASE") or (bdata.columns[1] == "phase") or (bdata.columns[1] == "Phase")


def unique(df: pd.DataFrame, colid: str) -> list:
    """Compatibility wrapper: ordered unique for DataFrames."""
    return _unique_df(df, colid)


def mean(X: np.ndarray, axis: int) -> np.ndarray:
    """Legacy mean helper that ignores NaNs and returns a 1D array."""
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map = X_nan_map.astype(int)
        X_[X_nan_map == 1] = 0

        # Calculate mean without accounting for NaN's
        if axis == 0:
            aux = np.sum(X_nan_map, axis=0)
            x_mean = np.sum(X_, axis=0, keepdims=True) / (
                (np.ones((1, X_.shape[1])) * X_.shape[0]) - aux
            )
        else:
            aux = np.sum(X_nan_map, axis=1)
            x_mean = np.sum(X_, axis=1, keepdims=True) / (
                (np.ones((X_.shape[0], 1)) * X_.shape[1]) - aux.reshape(-1, 1)
            )
    else:
        x_mean = np.mean(X_, axis=axis, keepdims=True)
    return x_mean.reshape(-1)


def simple_align(bdata: pd.DataFrame, nsamples: int) -> pd.DataFrame:
    """Simple alignment for batch data using row number to linearly interpolate."""
    bdata_a: list[pd.DataFrame] = []
    phase = _has_phase_col(bdata)
    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()
    for b in unique_batches:
        data_ = bdata[bdata[bdata.columns[0]] == b]
        indx = np.arange(data_.shape[0])
        new_indx = np.linspace(0, data_.shape[0] - 1, nsamples)
        bname_rs = [data_[bdata.columns[0]].values[0]] * nsamples
        if phase:
            phase_list = data_[bdata.columns[1]].values.tolist()
            roundindx = np.round(new_indx).astype("int").tolist()
            phase_rs = [phase_list[i] for i in roundindx]
            vals = data_.values[:, 2:]
            cols = data_.columns[2:]
        else:
            vals = data_.values[:, 1:]
            cols = data_.columns[1:]
        vals_rs = np.zeros((nsamples, vals.shape[1]))
        for i in np.arange(vals.shape[1]):
            vals_rs[:, i] = np.interp(new_indx, indx.astype("float"), vals[:, i].astype("float"))

        df_ = pd.DataFrame(vals_rs, columns=cols)
        if phase:
            df_.insert(0, bdata.columns[1], phase_rs)
        df_.insert(0, bdata.columns[0], bname_rs)
        bdata_a.append(df_)
    return pd.concat(bdata_a)


def phase_simple_align(bdata: pd.DataFrame, nsamples: dict[str, int]) -> pd.DataFrame:
    """Simple batch alignment (0 to 100%) per phase."""
    bdata_a: list[pd.DataFrame] = []
    phase = _has_phase_col(bdata)
    if not phase:
        raise ValueError("phase_simple_align requires a PHASE/Phase/phase column as column[1].")

    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()

    for b in unique_batches:
        data_ = bdata[bdata[bdata.columns[0]] == b]
        vals_rs: np.ndarray | list[Any] = []
        bname_rs: list[Any] = []
        phase_rs: list[Any] = []
        firstone = True
        for p in nsamples.keys():
            p_data = data_[data_[data_.columns[1]] == p]
            samps = nsamples[p]

            indx = np.arange(p_data.shape[0])
            new_indx = np.linspace(0, p_data.shape[0] - 1, samps)
            bname_rs_ = [p_data[p_data.columns[0]].values[0]] * samps
            phase_rs_ = [p_data[p_data.columns[1]].values[0]] * samps

            vals = p_data.values[:, 2:]
            cols = p_data.columns[2:]
            vals_rs_ = np.zeros((samps, vals.shape[1]))
            for i in np.arange(vals.shape[1]):
                vals_rs_[:, i] = np.interp(new_indx, indx.astype("float"), vals[:, i].astype("float"))
            if firstone:
                vals_rs = vals_rs_
                firstone = False
            else:
                vals_rs = np.vstack((vals_rs, vals_rs_))  # type: ignore[arg-type]
            bname_rs.extend(bname_rs_)
            phase_rs.extend(phase_rs_)

        df_ = pd.DataFrame(vals_rs, columns=cols)  # type: ignore[arg-type]
        df_.insert(0, bdata.columns[1], phase_rs)
        df_.insert(0, bdata.columns[0], bname_rs)
        bdata_a.append(df_)
    return pd.concat(bdata_a)


def phase_iv_align(bdata: pd.DataFrame, nsamples: dict[str, int | list]) -> pd.DataFrame:
    """Batch alignment using an indicator variable (IV) per phase (optional)."""
    bdata_a: list[pd.DataFrame] = []
    phase = _has_phase_col(bdata)
    if not phase:
        raise ValueError("phase_iv_align requires a PHASE/Phase/phase column as column[1].")

    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()

    for b in unique_batches:
        data_ = bdata[bdata[bdata.columns[0]] == b]
        vals_rs: np.ndarray | list[Any] = []
        bname_rs: list[Any] = []
        phase_rs: list[Any] = []
        firstone = True
        for p in nsamples.keys():
            p_data = data_[data_[data_.columns[1]] == p]
            samps = nsamples[p]

            if not isinstance(samps, list):  # Linear alignment
                indx = np.arange(p_data.shape[0])
                new_indx = np.linspace(0, p_data.shape[0] - 1, samps)
                bname_rs_ = [p_data[p_data.columns[0]].values[0]] * samps
                phase_rs_ = [p_data[p_data.columns[1]].values[0]] * samps

                vals = p_data.values[:, 2:]
                cols = p_data.columns[2:]
                vals_rs_ = np.zeros((samps, vals.shape[1]))
                for i in np.arange(vals.shape[1]):
                    x_ = indx.astype("float")
                    y_ = vals[:, i].astype("float")
                    x_ = x_[~np.isnan(y_)]
                    y_ = y_[~np.isnan(y_)]
                    if len(y_) == 0:
                        vals_rs_[:, i] = np.nan
                    else:
                        vals_rs_[:, i] = np.interp(new_indx, x_, y_, left=np.nan, right=np.nan)

            else:  # align this phase with IV
                if len(samps) == 4:
                    iv_id = samps[0]
                    nsamp = samps[1]
                    start = samps[2]
                    end = samps[3]
                    new_indx = np.linspace(start, end, nsamp)
                    iv = p_data[iv_id].values.astype(float)
                elif len(samps) == 3:
                    iv_id = samps[0]
                    nsamp = samps[1]
                    end = samps[2]

                    if (not (isinstance(end, int) or isinstance(end, float))) and isinstance(end, dict) and (not firstone):
                        # end is a model to estimate the end value based on previous trajectory
                        cols = p_data.columns[2:].tolist()
                        fakebatch = pd.DataFrame(vals_rs, columns=cols)  # type: ignore[arg-type]
                        fakebatch.insert(0, "phase", [p] * fakebatch.shape[0])
                        fakebatch.insert(0, "BatchID", [b] * fakebatch.shape[0])
                        preds = predict(fakebatch, end)  # type: ignore[arg-type]
                        end_hat = preds["Yhat"][preds["Yhat"].columns[1]].values[0]
                        iv = p_data[iv_id].values.astype(float)
                        end_hat = end_hat + iv[0]
                        new_indx = np.linspace(iv[0], end_hat, nsamp)
                    else:
                        iv = p_data[iv_id].values.astype(float)
                        new_indx = np.linspace(iv[0], end, nsamp)  # type: ignore[arg-type]
                else:
                    raise ValueError("IV phase spec must be [IVarID, nsamp, end] or [IVarID, nsamp, start, end].")

                cols = p_data.columns[2:].tolist()
                bname_rs_ = [p_data[p_data.columns[0]].values[0]] * nsamp
                phase_rs_ = [p_data[p_data.columns[1]].values[0]] * nsamp

                # Check monotonicity
                if np.all(np.diff(iv) < 0) or np.all(np.diff(iv) > 0):
                    indx = iv
                    vals = p_data.values[:, 2:]
                    vals_rs_ = np.zeros((nsamp, vals.shape[1]))
                    for i in np.arange(vals.shape[1]):
                        x_ = indx.astype("float")
                        y_ = vals[:, i].astype("float")
                        x_ = x_[~np.isnan(y_)]
                        y_ = y_[~np.isnan(y_)]
                        if len(y_) == 0:
                            vals_rs_[:, i] = np.nan
                        else:
                            vals_rs_[:, i] = np.interp(new_indx, x_, y_, left=np.nan, right=np.nan)
                else:
                    # if monotonicity fails try to remove non-monotonic vals
                    print(f"Indicator variable {iv_id} for batch {b} is not monotinic")
                    print("this is not ideal, maybe rethink your IV")
                    print("trying to remove non-monotonic samples")
                    x = np.arange(0, len(iv))
                    m = (1 / sum(x**2)) * sum(x * (iv - iv[0]))
                    expected_direction = -1 if m < 0 else 1
                    vals = p_data.values[:, 2:]
                    new_vals = vals[0, :]
                    prev_iv = iv[0]
                    mon_iv = [iv[0]]
                    for i in np.arange(1, vals.shape[0]):
                        if (np.sign(iv[i] - prev_iv) * expected_direction) == 1:
                            new_vals = np.vstack((new_vals, vals[i, :]))
                            prev_iv = iv[i]
                            mon_iv.append(iv[i])
                    indx = np.array(mon_iv)
                    vals = new_vals.copy()
                    vals_rs_ = np.zeros((nsamp, vals.shape[1]))
                    for i in np.arange(vals.shape[1]):
                        x_ = indx.astype("float")
                        y_ = vals[:, i].astype("float")
                        x_ = x_[~np.isnan(y_)]
                        y_ = y_[~np.isnan(y_)]
                        if len(y_) == 0:
                            vals_rs_[:, i] = np.nan
                        else:
                            vals_rs_[:, i] = np.interp(new_indx, x_, y_, left=np.nan, right=np.nan)

            if firstone:
                vals_rs = vals_rs_
                firstone = False
            else:
                vals_rs = np.vstack((vals_rs, vals_rs_))  # type: ignore[arg-type]
            bname_rs.extend(bname_rs_)
            phase_rs.extend(phase_rs_)

        df_ = pd.DataFrame(vals_rs, columns=cols)  # type: ignore[arg-type]
        df_.insert(0, bdata.columns[1], phase_rs)
        df_.insert(0, bdata.columns[0], bname_rs)
        bdata_a.append(df_)
    return pd.concat(bdata_a)


def plot_var_all_batches(
    bdata: pd.DataFrame,
    *,
    which_var: str | list[str] | bool = False,
    plot_title: str = "",
    mkr_style: str = ".-",
    phase_samples: dict | bool = False,
    alpha_: float = 0.2,
    timecolumn: str | bool = False,
    lot_legend: bool = False,
) -> None:
    """Plot batch data for all batches in a dataset."""
    var_list: str | list[str] | bool = which_var
    phase = _has_phase_col(bdata)
    if isinstance(var_list, bool):
        var_list = bdata.columns[2:].tolist() if phase else bdata.columns[1:].tolist()
    if isinstance(var_list, str) and not isinstance(var_list, list):
        var_list = [var_list]

    for v in var_list:
        plt.figure()
        if not isinstance(timecolumn, bool):
            dat = bdata[[bdata.columns[0], timecolumn, v]]
        else:
            dat = bdata[[bdata.columns[0], v]]
        for b in np.unique(dat[bdata.columns[0]]):
            data_ = dat[v][dat[dat.columns[0]] == b]
            if not isinstance(timecolumn, bool):
                tim = dat[timecolumn][dat[dat.columns[0]] == b]
                if lot_legend:
                    plt.plot(tim.values, data_.values, mkr_style, label=b)
                else:
                    plt.plot(tim.values, data_.values, mkr_style)
                plt.xlabel(timecolumn)
            else:
                x = 1 + np.arange(len(data_.values))
                if lot_legend:
                    plt.plot(x, data_.values, mkr_style, label=b)
                else:
                    plt.plot(x, data_.values, mkr_style)
                plt.xlabel("Sample")
            plt.ylabel(v)
        if lot_legend:
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        if not isinstance(phase_samples, bool):
            s_txt = 1
            s_lin = 1
            plt.axvline(x=1, color="magenta", alpha=alpha_)
            for p in phase_samples.keys():
                s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                plt.axvline(x=s_lin, color="magenta", alpha=alpha_)
                ylim_ = plt.ylim()
                plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
        plt.title(plot_title)
        plt.tight_layout()


def plot_batch(
    bdata: pd.DataFrame,
    which_batch: str | list[str],
    which_var: str | list[str],
    *,
    include_mean_exc: bool = False,
    include_set: bool = False,
    phase_samples: dict | bool = False,
    single_plot: bool = False,
    plot_title: str = "",
) -> None:
    """Plot one or more variables for one or more batches."""
    if isinstance(which_batch, str) and not isinstance(which_batch, list):
        which_batch = [which_batch]
    if isinstance(which_var, str) and not isinstance(which_var, list):
        which_var = [which_var]
    if single_plot:
        plt.figure()
        first_pass = True

    for b in which_batch:
        this_batch = bdata[bdata[bdata.columns[0]] == b]
        all_others = bdata[bdata[bdata.columns[0]] != b]
        for v in which_var:
            this_var_this_batch = this_batch[v].values
            if not single_plot:
                plt.figure()
            x_axis = np.arange(len(this_var_this_batch)) + 1
            plt.plot(x_axis, this_var_this_batch, "k", label=b)
            if include_mean_exc or include_set:
                this_var_all_others: list[np.ndarray] = []
                max_obs = 0
                for bb in np.unique(all_others[bdata.columns[0]]):
                    arr = all_others[v][all_others[bdata.columns[0]] == bb].values
                    this_var_all_others.append(arr)
                    max_obs = max(max_obs, len(arr))
                this_var_all_others_ = []
                for bb in np.unique(all_others[bdata.columns[0]]):
                    _to_append = all_others[v][all_others[bdata.columns[0]] == bb].values
                    this_len = len(_to_append)
                    _to_append = np.append(_to_append, np.tile(np.nan, max_obs - this_len))
                    this_var_all_others_.append(_to_append)
                this_var_all_others_ = np.array(this_var_all_others_)
                x_axis = np.arange(max_obs) + 1
                if not single_plot:
                    if include_set:
                        for t in this_var_all_others:
                            x_axis = np.arange(len(t)) + 1
                            plt.plot(x_axis, t, "m", alpha=0.1)
                        plt.plot(x_axis, this_var_all_others[-1], "m", alpha=0.1, label="rest of set")
                    if include_mean_exc:
                        plt.plot(x_axis, mean(this_var_all_others_, axis=0), "r", label="Mean without " + b)
                else:
                    if first_pass:
                        if include_set:
                            plt.plot(x_axis, this_var_all_others_.T, "m", alpha=0.1)
                            plt.plot(x_axis, this_var_all_others_[0, :], "m", alpha=0.1, label="rest of set")
                        if include_mean_exc:
                            plt.plot(x_axis, mean(this_var_all_others_, axis=0), "r", label="Mean without " + b)
                        first_pass = False
            if not isinstance(phase_samples, bool):
                s_txt = 1
                s_lin = 1
                plt.axvline(x=1, color="magenta", alpha=0.2)
                for p in phase_samples.keys():
                    s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                    plt.axvline(x=s_lin, color="magenta", alpha=0.2)
                    ylim_ = plt.ylim()
                    plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                    s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
            plt.title(b + " " + plot_title)
            plt.xlabel("sample")
            plt.ylabel(v)
            plt.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
            plt.tight_layout()


def unfold_horizontal(bdata: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Unfold vertically concatenated batch trajectories into one row per batch."""
    phase = _has_phase_col(bdata)

    firstone = True
    clbl: list[str] = []
    bid: list[str] = []
    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()
    for b in unique_batches:
        data_ = bdata[bdata[bdata.columns[0]] == b]
        if phase:
            vals = data_.values[:, 2:]
            cols = data_.columns[2:]
        else:
            vals = data_.values[:, 1:]
            cols = data_.columns[1:]
        row_: np.ndarray | list[Any] = []
        firstone_c = True
        for c in np.arange(vals.shape[1]):
            r_ = vals[:, [c]].reshape(1, -1)
            if firstone:
                for i in np.arange(1, r_.shape[1] + 1):
                    clbl.append(cols[c] + "_" + str(i))
                    bid.append(cols[c])
            if firstone_c:
                row_ = r_
                firstone_c = False
            else:
                row_ = np.hstack((row_, r_))  # type: ignore[arg-type]
        if firstone:
            bdata_hor = row_  # type: ignore[assignment]
            firstone = False
        else:
            bdata_hor = np.vstack((bdata_hor, row_))  # type: ignore[arg-type]
    bdata_hor_pd = pd.DataFrame(bdata_hor, columns=clbl)  # type: ignore[arg-type]
    bdata_hor_pd.insert(0, bdata.columns[0], unique_batches)
    return bdata_hor_pd, clbl, bid


def refold_horizontal(xuf: np.ndarray, nvars: int, nsamples: int) -> np.ndarray:
    """Refold a horizontally unfolded batch vector back to (samples x vars) trajectory per batch."""
    Xb: np.ndarray | list[Any] = []
    for i in np.arange(xuf.shape[0]):
        r = xuf[i, :]
        for v in np.arange(nvars):
            var = r[:nsamples]
            var = var.reshape(-1, 1)
            if v < (nvars - 1):
                r = r[nsamples:]
            if v == 0:
                batch = var
            else:
                batch = np.hstack((batch, var))
        if i == 0:
            Xb = batch
        else:
            Xb = np.vstack((Xb, batch))  # type: ignore[arg-type]
    return Xb  # type: ignore[return-value]


def _uf_l(L: np.ndarray, spb: int, vpb: int) -> np.ndarray:
    first_ = True
    for i in np.arange(L.shape[1]):
        col_ = L[:, [i]]
        s_ = np.arange(spb)
        first_flag = True
        for v in np.arange(vpb):
            s_2use = s_ + v * spb
            if first_flag:
                col_rearranged = col_[s_2use]
                first_flag = False
            else:
                col_rearranged = np.hstack((col_rearranged, col_[s_2use]))
        col_rearranged = col_rearranged.reshape(1, -1)
        if first_:
            L_uf_hor_mon = col_rearranged.T
            first_ = False
        else:
            L_uf_hor_mon = np.hstack((L_uf_hor_mon, col_rearranged.T))
    return L_uf_hor_mon


def _uf_hor_mon_loadings(mvmobj: dict) -> dict:
    spb = mvmobj["nsamples"]
    vpb = mvmobj["nvars"]
    is_pls = "Q" in mvmobj
    if is_pls:
        ninit = mvmobj["ninit"]

    if is_pls:
        if ninit > 0:
            z_ws = mvmobj["Ws"][:ninit, :]
            z_w = mvmobj["W"][:ninit, :]
            z_p = mvmobj["P"][:ninit, :]
            z_mx = mvmobj["mx"][:ninit]
            z_sx = mvmobj["sx"][:ninit]

            x_ws = mvmobj["Ws"][ninit:, :]
            x_w = mvmobj["W"][ninit:, :]
            x_p = mvmobj["P"][ninit:, :]
            x_mx = mvmobj["mx"][ninit:]
            x_sx = mvmobj["sx"][ninit:]
            Ws_ufm = np.vstack((z_ws, _uf_l(x_ws, spb, vpb)))
            W_ufm = np.vstack((z_w, _uf_l(x_w, spb, vpb)))
            P_ufm = np.vstack((z_p, _uf_l(x_p, spb, vpb)))
            mx_ufm = np.vstack(
                (z_mx.reshape(-1, 1), _uf_l(x_mx.reshape(-1, 1), spb, vpb))
            ).reshape(-1)
            sx_ufm = np.vstack(
                (z_sx.reshape(-1, 1), _uf_l(x_sx.reshape(-1, 1), spb, vpb))
            ).reshape(-1)
        else:
            Ws_ufm = _uf_l(mvmobj["Ws"], spb, vpb)
            W_ufm = _uf_l(mvmobj["W"], spb, vpb)
            P_ufm = _uf_l(mvmobj["P"], spb, vpb)
            mx_ufm = _uf_l(mvmobj["mx"].reshape(-1, 1), spb, vpb).reshape(-1)
            sx_ufm = _uf_l(mvmobj["sx"].reshape(-1, 1), spb, vpb).reshape(-1)
    else:
        P_ufm = _uf_l(mvmobj["P"], spb, vpb)
        mx_ufm = _uf_l(mvmobj["mx"].reshape(-1, 1), spb, vpb).reshape(-1)
        sx_ufm = _uf_l(mvmobj["sx"].reshape(-1, 1), spb, vpb).reshape(-1)

    if is_pls:
        mvmobj["Ws_ufm"] = Ws_ufm
        mvmobj["W_ufm"] = W_ufm
        mvmobj["P_ufm"] = P_ufm
    else:
        mvmobj["P_ufm"] = P_ufm

    mvmobj["mx_ufm"] = mx_ufm
    mvmobj["sx_ufm"] = sx_ufm
    return mvmobj


def loadings(mmvm_obj: dict, dim: int, *, r2_weighted: bool = False, which_var: str | list[str] | bool = False) -> None:
    """Plot batch loadings for variables as a function of time/sample."""
    dim = dim - 1
    if "Q" in mmvm_obj:
        if mmvm_obj["ninit"] == 0:
            aux_df = pd.DataFrame(mmvm_obj["Ws"] * mmvm_obj["r2xpv"] if r2_weighted else mmvm_obj["Ws"])
            aux_df.insert(0, "bid", mmvm_obj["bid"])
            if isinstance(which_var, bool):
                vars_to_plot = unique(aux_df, "bid")
            else:
                vars_to_plot = [which_var] if isinstance(which_var, str) else which_var
            for v in vars_to_plot:
                plt.figure()
                dat = aux_df[dim][aux_df["bid"] == v].values
                plt.fill_between(np.arange(mmvm_obj["nsamples"]) + 1, dat)
                plt.xlabel("sample")
                plt.ylabel(("$W^* * R^2$" if r2_weighted else "$W^*$") + f" [{dim+1}]")
                plt.title(v)
                plt.ylim(mmvm_obj["Ws"][:, dim].min() * 1.2, mmvm_obj["Ws"][:, dim].max() * 1.2)
                ylim_ = plt.ylim()
                phase_samples = mmvm_obj.get("phase_samples", False)
                if not isinstance(phase_samples, bool):
                    s_txt = 1
                    s_lin = 1
                    plt.axvline(x=1, color="magenta", alpha=0.2)
                    for p in phase_samples.keys():
                        s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                        plt.axvline(x=s_lin, color="magenta", alpha=0.2)
                        plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                        s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                plt.tight_layout()
        else:
            z_loadings = mmvm_obj["Ws"][np.arange(mmvm_obj["ninit"])]
            r2pvz = mmvm_obj["r2xpv"][np.arange(mmvm_obj["ninit"]), :]
            if r2_weighted:
                z_loadings = z_loadings * r2pvz

            zvars = mmvm_obj["varidX"][0 : mmvm_obj["ninit"]]
            plt.figure()
            plt.bar(zvars, z_loadings[:, dim])
            plt.xticks(rotation=90)
            plt.ylabel(("$W^* * R^2$" if r2_weighted else "$W^*$") + f" [{dim+1}]")
            plt.title("Loadings for Initial Conditions")
            plt.tight_layout()

            rows_ = np.arange(mmvm_obj["nsamples"] * mmvm_obj["nvars"]) + mmvm_obj["ninit"]
            aux_df = pd.DataFrame(
                mmvm_obj["Ws"][rows_, :] * mmvm_obj["r2xpv"][rows_, :] if r2_weighted else mmvm_obj["Ws"][rows_, :]
            )
            aux_df.insert(0, "bid", mmvm_obj["bid"])
            if isinstance(which_var, bool):
                vars_to_plot = unique(aux_df, "bid")
            else:
                vars_to_plot = [which_var] if isinstance(which_var, str) else which_var
            for v in vars_to_plot:
                plt.figure()
                dat = aux_df[dim][aux_df["bid"] == v].values
                plt.fill_between(np.arange(mmvm_obj["nsamples"]) + 1, dat)
                plt.xlabel("sample")
                plt.ylabel(("$W^* * R^2$" if r2_weighted else "$W^*$") + f" [{dim+1}]")
                plt.title(v)
                plt.ylim(mmvm_obj["Ws"][:, dim].min() * 1.2, mmvm_obj["Ws"][:, dim].max() * 1.2)
                ylim_ = plt.ylim()
                phase_samples = mmvm_obj.get("phase_samples", False)
                if not isinstance(phase_samples, bool):
                    s_txt = 1
                    s_lin = 1
                    plt.axvline(x=1, color="magenta", alpha=0.2)
                    for p in phase_samples.keys():
                        s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                        plt.axvline(x=s_lin, color="magenta", alpha=0.2)
                        plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                        s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                plt.tight_layout()
    else:
        aux_df = pd.DataFrame(mmvm_obj["P"] * mmvm_obj["r2xpv"] if r2_weighted else mmvm_obj["P"])
        aux_df.insert(0, "bid", mmvm_obj["bid"])
        if isinstance(which_var, bool):
            vars_to_plot = unique(aux_df, "bid")
        else:
            vars_to_plot = [which_var] if isinstance(which_var, str) else which_var
        for v in vars_to_plot:
            plt.figure()
            dat = aux_df[dim][aux_df["bid"] == v].values
            plt.fill_between(np.arange(mmvm_obj["nsamples"]) + 1, dat)
            plt.xlabel("sample")
            plt.ylabel(("P * $R^2$" if r2_weighted else "P") + f" [{dim+1}]")
            plt.title(v)
            plt.ylim(mmvm_obj["P"][:, dim].min() * 1.2, mmvm_obj["P"][:, dim].max() * 1.2)
            ylim_ = plt.ylim()
            phase_samples = mmvm_obj.get("phase_samples", False)
            if not isinstance(phase_samples, bool):
                s_txt = 1
                s_lin = 1
                plt.axvline(x=1, color="magenta", alpha=0.2)
                for p in phase_samples.keys():
                    s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                    plt.axvline(x=s_lin, color="magenta", alpha=0.2)
                    plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                    s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
            plt.tight_layout()


def loadings_abs_integral(mmvm_obj: dict, *, r2_weighted: bool = False, addtitle: str | bool = False) -> None:
    """Plot the integral of the absolute value of loadings for a batch model."""
    if "Q" in mmvm_obj:
        if mmvm_obj["ninit"] == 0:
            aux_df = pd.DataFrame(mmvm_obj["Ws"] * mmvm_obj["r2xpv"] if r2_weighted else mmvm_obj["Ws"])
            aux_df.insert(0, "bid", mmvm_obj["bid"])
        else:
            rows_ = np.arange(mmvm_obj["nsamples"] * mmvm_obj["nvars"]) + mmvm_obj["ninit"]
            aux_df = pd.DataFrame(
                mmvm_obj["Ws"][rows_, :] * mmvm_obj["r2xpv"][rows_, :] if r2_weighted else mmvm_obj["Ws"][rows_, :]
            )
            aux_df.insert(0, "bid", mmvm_obj["bid"])

        integral_of_loadings = []
        aux_vname = []
        for v in unique(aux_df, "bid"):
            dat = aux_df[aux_df["bid"] == v].values[:, 1:]
            integral_of_loadings.append(np.sum(np.abs(dat), axis=0, keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings = np.array(integral_of_loadings)
        for a in np.arange(mmvm_obj["T"].shape[1]):
            plt.figure()
            plt.bar(aux_vname, integral_of_loadings[:, a])
            plt.ylabel(r"$\sum (|W^*| * R^2$ [" + str(a + 1) + "])" if r2_weighted else r"$\sum (|W^*|$ [" + str(a + 1) + "])")
            plt.xticks(rotation=75)
            if not isinstance(addtitle, bool) and isinstance(addtitle, str):
                plt.title(addtitle)
            plt.tight_layout()
    else:
        aux_df = pd.DataFrame(mmvm_obj["P"] * mmvm_obj["r2xpv"] if r2_weighted else mmvm_obj["P"])
        aux_df.insert(0, "bid", mmvm_obj["bid"])
        integral_of_loadings = []
        aux_vname = []
        for v in unique(aux_df, "bid"):
            dat = aux_df[aux_df["bid"] == v].values[:, 1:]
            integral_of_loadings.append(np.sum(np.abs(dat), axis=0, keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings = np.array(integral_of_loadings)
        for a in np.arange(mmvm_obj["T"].shape[1]):
            plt.figure()
            plt.bar(aux_vname, integral_of_loadings[:, a])
            plt.ylabel(r"$\sum (|P| * R^2$ [" + str(a + 1) + "])" if r2_weighted else r"$\sum (|P| [" + str(a + 1) + "])")
            plt.xticks(rotation=75)
            if not isinstance(addtitle, bool) and isinstance(addtitle, str):
                plt.title(addtitle)
            plt.tight_layout()


def batch_vip(mmvm_obj: dict, *, addtitle: str | bool = False) -> None:
    """Batch-VIP-like plot (sum of |loadings| weighted by R² contributions)."""
    r2_weighted = True
    if "Q" in mmvm_obj:
        if mmvm_obj["ninit"] == 0:
            aux_df = pd.DataFrame(mmvm_obj["Ws"] * np.tile(mmvm_obj["r2y"], (mmvm_obj["Ws"].shape[0], 1)) if r2_weighted else mmvm_obj["Ws"])
            aux_df.insert(0, "bid", mmvm_obj["bid"])
        else:
            rows_ = np.arange(mmvm_obj["nsamples"] * mmvm_obj["nvars"]) + mmvm_obj["ninit"]
            aux_df = pd.DataFrame(
                mmvm_obj["Ws"][rows_, :] * mmvm_obj["r2xpv"][rows_, :] if r2_weighted else mmvm_obj["Ws"][rows_, :]
            )
            aux_df.insert(0, "bid", mmvm_obj["bid"])

        integral_of_loadings = []
        aux_vname = []
        for v in unique(aux_df, "bid"):
            dat = aux_df[aux_df["bid"] == v].values[:, 1:]
            integral_of_loadings.append(np.sum(np.abs(dat), axis=0, keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings = np.array(integral_of_loadings)
        plt.figure()
        plt.bar(aux_vname, np.sum(integral_of_loadings, axis=1))
        plt.ylabel("Batch VIP")
        plt.xticks(rotation=75)
        if not isinstance(addtitle, bool) and isinstance(addtitle, str):
            plt.title(addtitle)
        plt.tight_layout()
    else:
        aux_df = pd.DataFrame(mmvm_obj["P"] * mmvm_obj["r2xpv"] if r2_weighted else mmvm_obj["P"])
        aux_df.insert(0, "bid", mmvm_obj["bid"])
        integral_of_loadings = []
        aux_vname = []
        for v in unique(aux_df, "bid"):
            dat = aux_df[aux_df["bid"] == v].values[:, 1:]
            integral_of_loadings.append(np.sum(np.abs(dat), axis=0, keepdims=True)[0])
            aux_vname.append(v)
        integral_of_loadings = np.array(integral_of_loadings)
        plt.figure()
        plt.bar(aux_vname, np.sum(integral_of_loadings, axis=1))
        plt.ylabel(r"$\sum_{a} \sum (|P| * R^2$")
        plt.xticks(rotation=75)
        if not isinstance(addtitle, bool) and isinstance(addtitle, str):
            plt.title(addtitle)
        plt.tight_layout()


def r2pv(mmvm_obj: dict, *, which_var: str | list[str] | bool = False) -> None:
    """Plot batch r2 for variables as a function of time/sample."""
    if mmvm_obj["ninit"] == 0:
        aux_df = pd.DataFrame(mmvm_obj["r2xpv"])
        aux_df.insert(0, "bid", mmvm_obj["bid"])
        if isinstance(which_var, bool):
            vars_to_plot = unique(aux_df, "bid")
        else:
            vars_to_plot = [which_var] if isinstance(which_var, str) else which_var
        for v in vars_to_plot:
            dat = aux_df[aux_df["bid"] == v].values * 100
            dat = dat[:, 1:].astype(float)
            dat = np.cumsum(dat, axis=1)
            dat = np.hstack((np.zeros((dat.shape[0], 1)), dat))
            plt.figure()
            for a in np.arange(mmvm_obj["A"]) + 1:
                plt.fill_between(np.arange(mmvm_obj["nsamples"]) + 1, dat[:, a], dat[:, a - 1], label="LV #" + str(a))
            plt.xlabel("sample")
            plt.ylabel("$R^2$pvX (%)")
            plt.legend()
            plt.title(v)
            plt.ylim(0, 100)
            ylim_ = plt.ylim()
            phase_samples = mmvm_obj.get("phase_samples", False)
            if not isinstance(phase_samples, bool):
                s_txt = 1
                s_lin = 1
                plt.axvline(x=1, color="black", alpha=0.2)
                for p in phase_samples.keys():
                    s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                    plt.axvline(x=s_lin, color="black", alpha=0.2)
                    plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="black")
                    s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
            plt.tight_layout()

        if "Q" in mmvm_obj:
            r2pvy = mmvm_obj["r2ypv"] * 100
            lbls = ["LV #" + str(a) for a in np.arange(1, mmvm_obj["A"] + 1)]
            r2pvy_pd = pd.DataFrame(r2pvy, index=mmvm_obj["varidY"], columns=lbls)
            fig1, ax1 = plt.subplots()
            r2pvy_pd.plot(kind="bar", stacked=True, ax=ax1)
            ax1.set_ylabel("$R^2$pvY")
            ax1.set_title("$R^2$ per LV for Y-Space")
            fig1.tight_layout()
    else:
        r2pvz = mmvm_obj["r2xpv"][np.arange(mmvm_obj["ninit"]), :] * 100
        zvars = mmvm_obj["varidX"][0 : mmvm_obj["ninit"]]
        lbls = ["LV #" + str(a) for a in np.arange(1, mmvm_obj["A"] + 1)]
        r2pvz_pd = pd.DataFrame(r2pvz, index=zvars, columns=lbls)
        fig2, ax2 = plt.subplots()
        r2pvz_pd.plot(kind="bar", stacked=True, ax=ax2)
        ax2.set_ylabel("$R^2$pvZ")
        ax2.set_title("$R^2$ Initial Conditions")
        fig2.tight_layout()

        rows_ = np.arange(mmvm_obj["nsamples"] * mmvm_obj["nvars"]) + mmvm_obj["ninit"]
        aux_df = pd.DataFrame(mmvm_obj["r2xpv"][rows_, :])
        aux_df.insert(0, "bid", mmvm_obj["bid"])
        if isinstance(which_var, bool):
            vars_to_plot = unique(aux_df, "bid")
        else:
            vars_to_plot = [which_var] if isinstance(which_var, str) else which_var
        for v in vars_to_plot:
            dat = aux_df[aux_df["bid"] == v].values * 100
            dat = dat[:, 1:].astype(float)
            dat = np.cumsum(dat, axis=1)
            dat = np.hstack((np.zeros((dat.shape[0], 1)), dat))
            plt.figure()
            for a in np.arange(mmvm_obj["A"]) + 1:
                plt.fill_between(np.arange(mmvm_obj["nsamples"]) + 1, dat[:, a], dat[:, a - 1], label="LV #" + str(a))
            plt.xlabel("sample")
            plt.ylabel("$R^2$pvX (%)")
            plt.legend()
            plt.title(v)
            plt.ylim(0, 100)
            ylim_ = plt.ylim()
            phase_samples = mmvm_obj.get("phase_samples", False)
            if not isinstance(phase_samples, bool):
                s_txt = 1
                s_lin = 1
                plt.axvline(x=1, color="black", alpha=0.2)
                for p in phase_samples.keys():
                    s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                    plt.axvline(x=s_lin, color="black", alpha=0.2)
                    plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="black")
                    s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
            plt.tight_layout()

        if "Q" in mmvm_obj:
            r2pvy = mmvm_obj["r2ypv"] * 100
            lbls = ["LV #" + str(a) for a in np.arange(1, mmvm_obj["A"] + 1)]
            r2pvy_pd = pd.DataFrame(r2pvy, index=mmvm_obj["varidY"], columns=lbls)
            fig1, ax1 = plt.subplots()
            r2pvy_pd.plot(kind="bar", stacked=True, ax=ax1)
            ax1.set_ylabel("$R^2$pvY")
            ax1.set_title("$R^2$ per LV for Y-Space")
            fig1.tight_layout()


def mpca(
    xbatch: pd.DataFrame,
    a: int,
    *,
    unfolding: str = "batch wise",
    phase_samples: dict | bool = False,
    cross_val: int = 0,
) -> dict:
    """Multi-way PCA for batch analysis."""
    nvars = xbatch.shape[1] - (2 if _has_phase_col(xbatch) else 1)
    nbatches = len(np.unique(xbatch[xbatch.columns[0]]))
    nsamples = xbatch.shape[0] / nbatches

    if unfolding == "batch wise":
        # remove low variance columns keeping record of the original order
        x_uf_, colnames, bid_o = unfold_horizontal(xbatch)
        x_uf, colsrem = clean_low_variances(x_uf_, shush=True)
        mx_rem = np.array(x_uf_[colsrem].mean().tolist()) if len(colsrem) > 0 else np.array([])
        mpca_obj = pca(x_uf, a, cross_val=cross_val)
        if len(colsrem) > 0:
            xtra_col = np.zeros((2 + 2 * a, 1))
            xtra_col[0] = 1
            xtra_cols = np.tile(xtra_col, (1, len(colsrem)))
            xtra_cols[1, :] = mx_rem
            aux = np.vstack((mpca_obj["sx"], mpca_obj["mx"], mpca_obj["P"].T, mpca_obj["r2xpv"].T))
            aux = np.hstack((aux, xtra_cols))
            all_cols = x_uf.columns[1:].tolist()
            all_cols.extend(colsrem)
            aux_pd = pd.DataFrame(aux, columns=all_cols)
            aux_pd = aux_pd[colnames]
            aux_new = aux_pd.values

            mpca_obj["sx"] = aux_new[0, :].reshape(1, -1)
            mpca_obj["mx"] = aux_new[1, :].reshape(1, -1)
            aux_new = aux_new[2:, :]
            mpca_obj["P"] = aux_new[0:a, :].T
            mpca_obj["r2xpv"] = aux_new[a:, :].T

        mpca_obj["varidX"] = colnames
        mpca_obj["bid"] = bid_o
        mpca_obj["uf"] = "batch wise"
        mpca_obj["phase_samples"] = phase_samples
        mpca_obj["nvars"] = int(nvars)
        mpca_obj["nbatches"] = int(nbatches)
        mpca_obj["nsamples"] = int(nsamples)
        mpca_obj["ninit"] = 0
        mpca_obj["A"] = a
    elif unfolding == "variable wise":
        xbatch_ = xbatch.copy()
        if _has_phase_col(xbatch_):
            xbatch_.drop(xbatch_.columns[1], axis=1, inplace=True)
        xbatch_, _colsrem = clean_low_variances(xbatch_)
        xbatch_, _rowsrem = _clean_empty_rows_matrix(xbatch_)
        mpca_obj = pca(xbatch_, a)
        mpca_obj["uf"] = "variable wise"
        mpca_obj["phase_samples"] = phase_samples
        mpca_obj["nvars"] = int(nvars)
        mpca_obj["nbatches"] = int(nbatches)
        mpca_obj["nsamples"] = int(nsamples)
        mpca_obj["ninit"] = 0
        mpca_obj["A"] = a
    else:
        mpca_obj = {}
    return mpca_obj


def _mimic_monitoring(
    mmvm_obj_f: dict,
    bdata: pd.DataFrame,
    which_batch: str,
    *,
    zinit: pd.DataFrame | bool = False,
    shush: bool = False,
    soft_sensor: str | list[str] | bool = False,
) -> dict:
    if ((not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] == 0) or (
        (isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0
    ):
        raise ValueError("Model and data do not correspond")

    T_mon: list[Any] = []
    ht2_mon: list[Any] = []
    spe_mon: list[Any] = []
    spei_mon: list[Any] = []
    cont_spei: list[Any] = []
    cont_spe: list[Any] = []
    cont_ht2: list[Any] = []
    forecast: list[pd.DataFrame] = []
    if "Q" in mmvm_obj_f:
        forecast_y_: list[np.ndarray] = []

    this_batch = bdata[bdata[bdata.columns[0]] == which_batch]

    colnames = this_batch.columns[2:].tolist() if _has_phase_col(bdata) else this_batch.columns[1:].tolist()

    ss_indx: list[int] = []
    # Make soft_sensor var all missing data
    if not isinstance(soft_sensor, bool):
        if isinstance(soft_sensor, list):
            for v in soft_sensor:
                if v not in colnames:
                    print(f"Soft sensor {v} is not a variable in this dataset")
                else:
                    ss_indx.append(colnames.index(v))
        elif isinstance(soft_sensor, str):
            if soft_sensor not in colnames:
                print(f"Soft sensor {soft_sensor} is not a variable in this dataset")
            else:
                ss_indx = [colnames.index(soft_sensor)]

    if not isinstance(zinit, bool):
        this_z = zinit[zinit[zinit.columns[0]] == which_batch]
        vals_z = this_z.values[:, 1:]
        cols_z = this_z.columns[1:]
        vals_z = np.array(vals_z, dtype=np.float64).reshape(-1)

    vals = this_batch.values[:, 2:] if _has_phase_col(bdata) else this_batch.values[:, 1:]
    vals = np.array(vals, dtype=np.float64)
    if len(ss_indx) > 0:
        vals[:, ss_indx] = np.nan

    if not shush:
        print("Running batch: " + which_batch)

    for k in np.arange(mmvm_obj_f["nsamples"]):
        x_uf_k = vals[: k + 1, :].reshape(1, -1)[0].tolist()
        num_nans = mmvm_obj_f["nvars"] * mmvm_obj_f["nsamples"] - len(x_uf_k)
        x_uf_k.extend([np.nan] * num_nans)
        x_uf_k = np.array(x_uf_k)

        if "Q" in mmvm_obj_f:
            if (not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0:
                x_2_pred = np.hstack((vals_z, x_uf_k))
                preds = pls_pred(x_2_pred, mmvm_obj_f)
            elif isinstance(zinit, bool) and mmvm_obj_f["ninit"] == 0:
                preds = pls_pred(x_uf_k, mmvm_obj_f)
            else:
                raise ValueError("Model and data do not correspond")
        else:
            preds = pca_pred(x_uf_k, mmvm_obj_f)

        T_mon.append(preds["Tnew"][0])
        ht2_mon.append(preds["T2"][0])

        if "Q" in mmvm_obj_f:
            forecast_y_.append(preds["Yhat"][0].reshape(-1))
            if (not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0:
                ninit = mmvm_obj_f["ninit"]
                preds_x = preds["Xhat"][0, ninit:]
                preds_z = preds["Xhat"][0, :ninit]
                aux_df = pd.DataFrame(preds_x.reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
                forecast.append(aux_df)
                inst_preds = preds_x.reshape(-1)
                inst_preds = (inst_preds - mmvm_obj_f["mx"][ninit:]) / mmvm_obj_f["sx"][ninit:]
                x_uf_k = (x_uf_k - mmvm_obj_f["mx"][ninit:]) / mmvm_obj_f["sx"][ninit:]
            else:
                aux_df = pd.DataFrame(preds["Xhat"].reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
                forecast.append(aux_df)
                inst_preds = preds["Xhat"].reshape(-1)
                inst_preds = (inst_preds - mmvm_obj_f["mx"]) / mmvm_obj_f["sx"]
                x_uf_k = (x_uf_k - mmvm_obj_f["mx"]) / mmvm_obj_f["sx"]
        else:
            aux_df = pd.DataFrame(preds["Xhat"].reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
            forecast.append(aux_df)
            inst_preds = preds["Xhat"].reshape(-1)
            inst_preds = (inst_preds - mmvm_obj_f["mx"]) / mmvm_obj_f["sx"]
            x_uf_k = (x_uf_k - mmvm_obj_f["mx"]) / mmvm_obj_f["sx"]

        inst_preds[np.isnan(x_uf_k)] = 0
        x_uf_k[np.isnan(x_uf_k)] = 0

        cont_ht2_ = np.zeros(len(x_uf_k) if (isinstance(zinit, bool) or mmvm_obj_f["ninit"] == 0) else len(x_2_pred))
        var_t = np.var(mmvm_obj_f["T"], ddof=1, axis=0)
        for a in np.arange(mmvm_obj_f["A"]):
            if "Q" in mmvm_obj_f:
                if (not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0:
                    x_2_pred_ = (x_2_pred - mmvm_obj_f["mx"]) / mmvm_obj_f["sx"]
                    cont_ht2_ += (x_2_pred_ * mmvm_obj_f["Ws"][:, a]) ** 2 / var_t[a]
                else:
                    cont_ht2_ += (x_uf_k * mmvm_obj_f["Ws"][:, a]) ** 2 / var_t[a]
            else:
                cont_ht2_ += (x_uf_k * mmvm_obj_f["P"][:, a]) ** 2 / var_t[a]

        if (not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0:
            ninit = mmvm_obj_f["ninit"]
            aux_df = pd.DataFrame(cont_ht2_[ninit:].reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
            cont_ht2.append(aux_df)
        else:
            aux_df = pd.DataFrame(cont_ht2_.reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
            cont_ht2.append(aux_df)

        spe_ = (inst_preds - x_uf_k) ** 2
        aux_df = pd.DataFrame(spe_.reshape(mmvm_obj_f["nsamples"], -1), columns=colnames)
        cont_spe.append(aux_df)
        spe_mon.append(np.sum(spe_))

        inst_samp = x_uf_k[k * mmvm_obj_f["nvars"] : (k + 1) * mmvm_obj_f["nvars"]]
        inst_preds_slice = inst_preds[k * mmvm_obj_f["nvars"] : (k + 1) * mmvm_obj_f["nvars"]]
        spei_ = (inst_preds_slice - inst_samp) ** 2
        cont_spei.append(spei_)
        spei_mon.append(np.sum(spei_))

    diags = {
        "Batch": which_batch,
        "t_mon": np.array(T_mon),
        "HT2_mon": np.array(ht2_mon),
        "spe_mon": np.array(spe_mon).reshape(-1),
        "cont_spe": cont_spe,
        "spei_mon": np.array(spei_mon),
        "cont_spei": pd.DataFrame(np.array(cont_spei), columns=colnames),
        "cont_ht2": cont_ht2,
        "forecast": forecast,
    }
    if "Q" in mmvm_obj_f:
        forecast_y = pd.DataFrame(np.array(forecast_y_), columns=mmvm_obj_f["varidY"])
        diags["forecast y"] = forecast_y
        if (not isinstance(zinit, bool)) and mmvm_obj_f["ninit"] > 0:
            ninit = mmvm_obj_f["ninit"]
            vals_z_sc = (vals_z - mmvm_obj_f["mx"][:ninit]) / mmvm_obj_f["sx"][:ninit]
            preds_z_sc = (preds_z - mmvm_obj_f["mx"][:ninit]) / mmvm_obj_f["sx"][:ninit]
            spe_z = (vals_z_sc - preds_z_sc) ** 2
            diags["reconstructed z"] = preds_z
            diags["spe z"] = float(np.sum(spe_z))
            diags["cont_spe_z"] = pd.DataFrame(spe_z, index=cols_z, columns=["Vars"])
            diags["cont_ht2_z"] = pd.DataFrame(cont_ht2_[:ninit], index=cols_z, columns=["Vars"])
    return diags


def monitor(
    mmvm_obj: dict,
    bdata: pd.DataFrame,
    *,
    which_batch: str | list[str] | bool = False,
    zinit: pd.DataFrame | bool = False,
    build_ci: bool = True,
    shush: bool = False,
    soft_sensor: str | list[str] | bool = False,
) -> dict | str | None:
    """Mimic real-time monitoring of a batch given a multi-way PCA/PLS model."""
    mmvm_obj_f = mmvm_obj.copy()
    mmvm_obj_f = _uf_hor_mon_loadings(mmvm_obj_f)
    mmvm_obj_f["mx"] = mmvm_obj_f["mx_ufm"]
    mmvm_obj_f["sx"] = mmvm_obj_f["sx_ufm"]
    mmvm_obj_f["P"] = mmvm_obj_f["P_ufm"]
    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()

    if "Q" in mmvm_obj:
        mmvm_obj_f["W"] = mmvm_obj_f["W_ufm"]
        mmvm_obj_f["Ws"] = mmvm_obj_f["Ws_ufm"]

    if isinstance(which_batch, bool) and build_ci:
        if mmvm_obj["nbatches"] == len(np.unique(bdata[bdata.columns[0]])):
            SPE: list[np.ndarray] = []
            SPEi: list[np.ndarray] = []
            T = np.zeros((mmvm_obj["nsamples"], mmvm_obj["A"], mmvm_obj["nbatches"]))
            if not shush:
                print("Building real_time confidence intervals")
            for i, b in enumerate(unique_batches):
                diags = _mimic_monitoring(mmvm_obj_f, bdata, b, zinit=zinit, soft_sensor=soft_sensor)
                SPE.append(diags["spe_mon"])
                SPEi.append(diags["spei_mon"])
                T[:, :, i] = diags["t_mon"]

            t_mon_ci_95 = []
            t_mon_ci_99 = []
            for a in np.arange(mmvm_obj["A"]):
                t_rt_ci_95 = []
                t_rt_ci_99 = []
                t = T[:, a, :].T
                for j in np.arange(mmvm_obj_f["nsamples"]):
                    l95, l99 = single_score_conf_int(t[:, [j]])
                    t_rt_ci_95.append(l95)
                    t_rt_ci_99.append(l99)
                t_mon_ci_95.append(np.array(t_rt_ci_95))
                t_mon_ci_99.append(np.array(t_rt_ci_99))

            n = mmvm_obj["nbatches"]
            A = mmvm_obj["A"]
            ht2_mon_ci_99 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f99(A, (n - A))
            ht2_mon_ci_95 = (((n - 1) * (n + 1) * A) / (n * (n - A))) * f95(A, (n - A))

            spe_mon_ci_95 = []
            spe_mon_ci_99 = []
            spei_mon_ci_95 = []
            spei_mon_ci_99 = []
            SPE_arr = np.array(SPE)
            SPEi_arr = np.array(SPEi)
            for j in np.arange(mmvm_obj_f["nsamples"]):
                l95, l99 = spe_ci(SPE_arr[:, [j]])
                spe_mon_ci_95.append(l95)
                spe_mon_ci_99.append(l99)
                l95, l99 = spe_ci(SPEi_arr[:, [j]])
                spei_mon_ci_95.append(l95)
                spei_mon_ci_99.append(l99)

            mmvm_obj["t_mon_ci_95"] = t_mon_ci_95
            mmvm_obj["t_mon_ci_99"] = t_mon_ci_99
            mmvm_obj["ht2_mon_ci_99"] = ht2_mon_ci_99
            mmvm_obj["ht2_mon_ci_95"] = ht2_mon_ci_95
            mmvm_obj["spe_mon_ci_95"] = spe_mon_ci_95
            mmvm_obj["spe_mon_ci_99"] = spe_mon_ci_99
            mmvm_obj["spei_mon_ci_95"] = spei_mon_ci_95
            mmvm_obj["spei_mon_ci_99"] = spei_mon_ci_99
            if not shush:
                print("Done")
            return None
        else:
            print("This ain't the data this model was trained on")
            return None
    else:
        has_ci = "t_mon_ci_95" in mmvm_obj
        if not has_ci:
            print("No monitoring conf. int. have been calculated")
            print('for this model object please run: "monitor(model_obj,training_data)" ')

        if isinstance(which_batch, str):
            which_batch = [which_batch]
        diags_list: list[dict] = []
        allok = True
        assert not isinstance(which_batch, bool)
        for b in which_batch:
            if b in bdata[bdata.columns[0]].values.tolist():
                diags_list.append(_mimic_monitoring(mmvm_obj_f, bdata, b, zinit=zinit, soft_sensor=soft_sensor))
            else:
                print("Batch not found in data set")
                allok = False
        if not allok:
            return "error batch not found"

        for a in np.arange(mmvm_obj["A"]):
            plt.figure()
            for i, b in enumerate(which_batch):
                x_axis = np.arange(len(diags_list[i]["t_mon"][:, [a]])) + 1
                plt.plot(x_axis, diags_list[i]["t_mon"][:, [a]], "o", label=b)
            if has_ci:
                plt.plot(x_axis, mmvm_obj["t_mon_ci_95"][a], "y", alpha=0.3)
                plt.plot(x_axis, -mmvm_obj["t_mon_ci_95"][a], "y", alpha=0.3)
                plt.plot(x_axis, mmvm_obj["t_mon_ci_99"][a], "r", alpha=0.3)
                plt.plot(x_axis, -mmvm_obj["t_mon_ci_99"][a], "r", alpha=0.3)
            plt.xlabel("sample")
            plt.ylabel("$t_" + str(a + 1) + "$")
            plt.title("Real time monitoring: Score plot $t_" + str(a + 1) + "$")
            box = plt.gca().get_position()
            plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.figure()
        for i, b in enumerate(which_batch):
            plt.plot(x_axis, diags_list[i]["HT2_mon"], "o", label=b)
            xlim_ = plt.xlim()
        if has_ci:
            plt.plot([0, xlim_[1]], [mmvm_obj["ht2_mon_ci_95"], mmvm_obj["ht2_mon_ci_95"]], "y", alpha=0.3)
            plt.plot([0, xlim_[1]], [mmvm_obj["ht2_mon_ci_99"], mmvm_obj["ht2_mon_ci_99"]], "r", alpha=0.3)
        plt.xlabel("sample")
        plt.ylabel("Hotelling's $T^2$")
        plt.title("Real time monitoring: Hotelling's $T^2$")
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.figure()
        for i, b in enumerate(which_batch):
            plt.plot(x_axis, diags_list[i]["spe_mon"], "o", label=b)
        if has_ci:
            plt.plot(x_axis, mmvm_obj["spe_mon_ci_95"], "y", alpha=0.3)
            plt.plot(x_axis, mmvm_obj["spe_mon_ci_99"], "r", alpha=0.3)
        plt.xlabel("sample")
        plt.ylabel("Global SPE")
        plt.title("Real time monitoring: Global SPE")
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        plt.figure()
        for i, b in enumerate(which_batch):
            plt.plot(x_axis, diags_list[i]["spei_mon"], "o", label=b)
        if has_ci:
            plt.plot(x_axis, mmvm_obj["spei_mon_ci_95"], "y", alpha=0.3)
            plt.plot(x_axis, mmvm_obj["spei_mon_ci_99"], "r", alpha=0.3)
        plt.xlabel("sample")
        plt.ylabel("Instantaneous SPE")
        plt.title("Real time monitoring: Instantaneous SPE")
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        if "Q" in mmvm_obj:
            for v in diags_list[0]["forecast y"].columns:
                plt.figure()
                for i, b in enumerate(which_batch):
                    plt.plot(x_axis, diags_list[i]["forecast y"][v], "o", label=b)
                plt.xlabel("sample")
                plt.ylabel(v)
                plt.title("Dynamic forecast of Y")

        return diags_list[0] if len(diags_list) == 1 else {"diagnostics": diags_list}


def mpls(
    xbatch: pd.DataFrame,
    y: pd.DataFrame,
    a: int,
    *,
    zinit: pd.DataFrame | bool = False,
    phase_samples: dict | bool = False,
    mb_each_var: bool = False,
    cross_val: int = 0,
    cross_val_X: bool = False,
) -> dict:
    """Multi-way PLS for batch analysis."""
    nvars = xbatch.shape[1] - (2 if _has_phase_col(xbatch) else 1)
    nbatches = len(np.unique(xbatch[xbatch.columns[0]]))
    nsamples = xbatch.shape[0] / nbatches

    x_uf_, colnames, bid_o = unfold_horizontal(xbatch)
    x_uf, colsrem = clean_low_variances(x_uf_, shush=True)
    mx_rem = np.array(x_uf_[colsrem].mean().tolist()) if len(colsrem) > 0 else np.array([])

    aux = np.array([colnames, bid_o])
    col_names_bid_pd = pd.DataFrame(aux.T, columns=["col name", "bid"])
    col_names_bid_pd_ = col_names_bid_pd[col_names_bid_pd["col name"].isin(x_uf.columns[1:].tolist())]
    unique_bid = col_names_bid_pd_.drop_duplicates(subset="bid", keep="first")["bid"].values.tolist()

    if not isinstance(zinit, bool):
        zinit, _rc = clean_low_variances(zinit)
        zcols = zinit.columns[1:].tolist()
        XMB: dict | pd.DataFrame = {"Initial Conditions": zinit}
    else:
        XMB = {}

    if mb_each_var:
        assert isinstance(XMB, dict)
        for v in unique_bid:
            these_cols = [x_uf.columns[0]]
            these_cols.extend(col_names_bid_pd_["col name"][col_names_bid_pd_["bid"] == v].values.tolist())
            varblock = x_uf[these_cols]
            XMB[v] = varblock
    else:
        if not isinstance(zinit, bool):
            assert isinstance(XMB, dict)
            XMB["Trajectories"] = x_uf
        else:
            XMB = x_uf

    if not isinstance(XMB, dict):
        mpls_obj = pls(XMB, y, a, cross_val=cross_val, cross_val_X=cross_val_X, force_nipals=True)
        yhat = pls_pred(XMB, mpls_obj)["Yhat"]
    else:
        mpls_obj = mbpls(XMB, y, a, cross_val_=cross_val, cross_val_X_=cross_val_X, force_nipals_=True)
        yhat = pls_pred(XMB, mpls_obj)["Yhat"]

    if len(colsrem) > 0:
        if not isinstance(zinit, bool):
            ninit_vars = zinit.shape[1] - 1
            z_sx = mpls_obj["sx"][np.arange(ninit_vars)].reshape(-1)
            z_mx = mpls_obj["mx"][np.arange(ninit_vars)].reshape(-1)
            z_ws = mpls_obj["Ws"][np.arange(ninit_vars), :]
            z_w = mpls_obj["W"][np.arange(ninit_vars), :]
            z_p = mpls_obj["P"][np.arange(ninit_vars), :]
            z_r2pv = mpls_obj["r2xpv"][np.arange(ninit_vars), :]
            xc_ = np.arange(ninit_vars, x_uf.shape[1] + ninit_vars - 1)
            xuf_sx = mpls_obj["sx"][xc_]
            xuf_mx = mpls_obj["mx"][xc_]
            xuf_ws = mpls_obj["Ws"][xc_, :]
            xuf_w = mpls_obj["W"][xc_, :]
            xuf_p = mpls_obj["P"][xc_, :]
            xuf_r2pv = mpls_obj["r2xpv"][xc_, :]
        else:
            xuf_sx = mpls_obj["sx"]
            xuf_mx = mpls_obj["mx"]
            xuf_ws = mpls_obj["Ws"]
            xuf_w = mpls_obj["W"]
            xuf_p = mpls_obj["P"]
            xuf_r2pv = mpls_obj["r2xpv"]

        xtra_col = np.zeros((2 + 4 * a, 1))
        xtra_col[0] = 1
        xtra_cols = np.tile(xtra_col, (1, len(colsrem)))
        xtra_cols[1, :] = mx_rem
        aux = np.vstack((xuf_sx, xuf_mx, xuf_ws.T, xuf_w.T, xuf_p.T, xuf_r2pv.T))
        aux = np.hstack((aux, xtra_cols))
        all_cols = x_uf.columns[1:].tolist()
        all_cols.extend(colsrem)
        aux_pd = pd.DataFrame(aux, columns=all_cols)
        aux_pd = aux_pd[colnames]
        aux_new = aux_pd.values

        if not isinstance(zinit, bool):
            mpls_obj["sx"] = np.hstack((z_sx, aux_new[0, :].reshape(-1)))
            mpls_obj["mx"] = np.hstack((z_mx, aux_new[1, :].reshape(-1)))
            aux_new2 = aux_new[2:, :]
            ws_ = aux_new2[0:a, :]
            mpls_obj["Ws"] = np.vstack((z_ws, ws_.T))
            aux_new2 = aux_new2[a:, :]
            w_ = aux_new2[0:a, :]
            mpls_obj["W"] = np.vstack((z_w, w_.T))
            aux_new2 = aux_new2[a:, :]
            p_ = aux_new2[0:a, :]
            mpls_obj["P"] = np.vstack((z_p, p_.T))
            aux_new2 = aux_new2[a:, :]
            r2xpv_ = aux_new2[0:a, :]
            mpls_obj["r2xpv"] = np.vstack((z_r2pv, r2xpv_.T))
            zcols.extend(colnames)
            colnames = zcols
        else:
            mpls_obj["sx"] = aux_new[0, :].reshape(1, -1)
            mpls_obj["mx"] = aux_new[1, :].reshape(1, -1)
            aux_new2 = aux_new[2:, :]
            ws_ = aux_new2[0:a, :]
            mpls_obj["Ws"] = ws_.T
            aux_new2 = aux_new2[a:, :]
            w_ = aux_new2[0:a, :]
            mpls_obj["W"] = w_.T
            aux_new2 = aux_new2[a:, :]
            p_ = aux_new2[0:a, :]
            mpls_obj["P"] = p_.T
            aux_new2 = aux_new2[a:, :]
            r2xpv_ = aux_new2[0:a, :]
            mpls_obj["r2xpv"] = r2xpv_.T

    mpls_obj["Yhat"] = yhat
    mpls_obj["varidX"] = colnames
    mpls_obj["bid"] = bid_o
    mpls_obj["uf"] = "batch wise"
    mpls_obj["nvars"] = int(nvars)
    mpls_obj["nbatches"] = int(nbatches)
    mpls_obj["nsamples"] = int(nsamples)
    mpls_obj["A"] = a
    mpls_obj["phase_samples"] = phase_samples
    mpls_obj["mb_each_var"] = mb_each_var
    mpls_obj["ninit"] = int(zinit.shape[1] - 1) if not isinstance(zinit, bool) else 0
    return mpls_obj


def phase_sampling_dist(
    bdata: pd.DataFrame, time_column: str | bool = False, addtitle: str | bool = False, use_phases: list[str] | bool = False
) -> dict | None:
    """Count and plot distribution of samples (or time) consumed per phase across batches."""
    data: dict[str, dict[str, float]] = {}
    if not _has_phase_col(bdata):
        print("Data is missing phase information or phase column is nor properly labeled")
        return None

    bids = unique(bdata, bdata.columns[0])
    phases = unique(bdata, bdata.columns[1]) if isinstance(use_phases, bool) else use_phases
    fig, ax = plt.subplots(1, len(phases) + 1)

    for i, p in enumerate(phases):
        totsamps = []
        samps_ = []
        samps_ind: dict[str, float] = {}
        for b in bids:
            bdat = bdata[(bdata[bdata.columns[1]] == p) & (bdata[bdata.columns[0]] == b)]
            if len(bdat) == 0:
                samps_.append(0)
                samps_ind[b] = 0
            elif not isinstance(time_column, bool):
                samps_.append(bdat[time_column].values[-1] - bdat[time_column].values[0])
                totsamps.append(bdata[time_column][(bdata[bdata.columns[0]] == b)].values[-1])
                samps_ind[b] = bdat[time_column].values[-1] - bdat[time_column].values[0]
            else:
                samps_.append(len(bdat))
                totsamps.append(len(bdata[(bdata[bdata.columns[0]] == b)]))
                samps_ind[b] = float(len(bdat))
        ax[i].hist(samps_)
        data[p] = samps_ind
        ax[i].set_xlabel(time_column if not isinstance(time_column, bool) else "# Samples")
        ax[i].set_ylabel("Count")
        ax[i].set_title(p)
    ax[-1].hist(totsamps)
    ax[-1].set_xlabel(time_column if not isinstance(time_column, bool) else "# Samples")
    ax[-1].set_ylabel("Count")
    ax[-1].set_title("Total")
    if not isinstance(addtitle, bool):
        fig.suptitle(addtitle)
    fig.tight_layout()
    return data


def predict(xbatch: pd.DataFrame, mmvm_obj: dict, *, zinit: pd.DataFrame | bool = False) -> dict:
    """Generate predictions for a Multi-way PCA/PLS model."""
    if "Q" in mmvm_obj:
        x_uf, colnames, bid_o = unfold_horizontal(xbatch)
        aux = np.array([colnames, bid_o])
        col_names_bid_pd = pd.DataFrame(aux.T, columns=["col name", "bid"])
        unique_bid = col_names_bid_pd.drop_duplicates(subset="bid", keep="first")["bid"].values.tolist()

        if not isinstance(zinit, bool):
            XMB: dict | pd.DataFrame = {"Initial Conditions": zinit}
        else:
            XMB = {}

        if mmvm_obj.get("mb_each_var", False):
            assert isinstance(XMB, dict)
            for v in unique_bid:
                these_cols = [x_uf.columns[0]]
                these_cols.extend(col_names_bid_pd["col name"][col_names_bid_pd["bid"] == v].values.tolist())
                varblock = x_uf[these_cols]
                XMB[v] = varblock
        else:
            if not isinstance(zinit, bool):
                assert isinstance(XMB, dict)
                XMB["Trajectories"] = x_uf
            else:
                XMB = x_uf

        pred = pls_pred(XMB, mmvm_obj)

        if not isinstance(zinit, bool):
            Zhat = pred["Xhat"][:, : mmvm_obj["ninit"]]
            Xhat = pred["Xhat"][:, mmvm_obj["ninit"] :]
            Xb = refold_horizontal(Xhat, mmvm_obj["nvars"], mmvm_obj["nsamples"])
            if _has_phase_col(xbatch):
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[2:])
                Xb.insert(0, xbatch.columns[1], xbatch[xbatch.columns[1]].values)
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            else:
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[1:])
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            pred["Xhat"] = Xb

            Zhat_df = pd.DataFrame(Zhat, columns=zinit.columns[1:].tolist())
            Zhat_df.insert(0, zinit.columns[0], zinit[zinit.columns[0]].values.astype(str).tolist())
            pred["Zhat"] = Zhat_df

            test = xbatch.drop_duplicates(subset=xbatch.columns[0], keep="first")
            Y_df = pd.DataFrame(pred["Yhat"], columns=mmvm_obj["varidY"])
            Y_df.insert(0, test.columns[0], test[test.columns[0]].values.astype(str).tolist())
            pred["Yhat"] = Y_df
        else:
            Xb = refold_horizontal(pred["Xhat"], mmvm_obj["nvars"], mmvm_obj["nsamples"])
            if _has_phase_col(xbatch):
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[2:])
                Xb.insert(0, xbatch.columns[1], xbatch[xbatch.columns[1]].values)
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            else:
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[1:])
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            pred["Xhat"] = Xb
            test = xbatch.drop_duplicates(subset=xbatch.columns[0], keep="first")
            Y_df = pd.DataFrame(pred["Yhat"], columns=mmvm_obj["varidY"])
            Y_df.insert(0, test.columns[0], test[test.columns[0]].values.astype(str).tolist())
            pred["Yhat"] = Y_df

    else:
        if mmvm_obj["uf"] == "batch wise":
            x_uf, _colnames, _bid_o = unfold_horizontal(xbatch)
            pred = pca_pred(x_uf, mmvm_obj)
            Xb = refold_horizontal(pred["Xhat"], mmvm_obj["nvars"], mmvm_obj["nsamples"])
            if _has_phase_col(xbatch):
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[2:])
                Xb.insert(0, xbatch.columns[1], xbatch[xbatch.columns[1]].values)
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            else:
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[1:])
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            pred["Xhat"] = Xb
        elif mmvm_obj["uf"] == "variable wise":
            xbatch_ = xbatch.copy()
            if _has_phase_col(xbatch_):
                xbatch_.drop(xbatch_.columns[1], axis=1, inplace=True)
            pred = pca_pred(xbatch_, mmvm_obj)
            Xb = pred["Xhat"]
            if _has_phase_col(xbatch):
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[2:])
                Xb.insert(0, xbatch.columns[1], xbatch[xbatch.columns[1]].values)
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            else:
                Xb = pd.DataFrame(Xb, columns=xbatch.columns[1:])
                Xb.insert(0, xbatch.columns[0], xbatch[xbatch.columns[0]].values)
            pred["Xhat"] = Xb
        else:
            pred = {}
    return pred


def _plot_contribs(
    bdata: pd.DataFrame,
    ylims: tuple[float, float],
    *,
    var_list: str | list[str] | bool = False,
    phase_samples: dict | bool = False,
    alpha_: float = 0.2,
    plot_title_: str = "",
) -> None:
    phase = _has_phase_col(bdata)
    if isinstance(var_list, bool):
        var_list = bdata.columns[2:].tolist() if phase else bdata.columns[1:].tolist()
    if isinstance(var_list, str) and not isinstance(var_list, list):
        var_list = [var_list]

    for v in var_list:
        plt.figure()
        dat = bdata[[bdata.columns[0], v]]
        for b in np.unique(dat[bdata.columns[0]]):
            data_ = dat[v][dat[dat.columns[0]] == b]
            plt.fill_between(np.arange(len(data_.values)) + 1, data_.values)
            plt.xlabel("Sample")
            plt.ylabel(v)
        if not isinstance(phase_samples, bool):
            s_txt = 1
            s_lin = 1
            plt.axvline(x=1, color="magenta", alpha=alpha_)
            for p in phase_samples.keys():
                s_lin += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
                plt.axvline(x=s_lin, color="magenta", alpha=alpha_)
                ylim_ = plt.ylim()
                plt.annotate(p, (s_txt, ylim_[0]), rotation=90, alpha=0.5, color="magenta")
                s_txt += phase_samples[p][1] if isinstance(phase_samples[p], list) else phase_samples[p]
        plt.ylim(ylims)
        plt.title(plot_title_)
        plt.tight_layout()


def contributions(
    mmvmobj: dict,
    X: pd.DataFrame,
    cont_type: str,
    *,
    to_obs: list[str] | bool = False,
    from_obs: list[str] | bool = False,
    lv_space: int | list[int] | bool = False,
    phase_samples: dict | bool = False,
    dyn_conts: bool = False,
    which_var: str | list[str] | bool = False,
    plot_title: str = "",
) -> np.ndarray | None:
    """Plot batch contribution plots to Scores, HT2 or SPE."""
    bvars = X.columns[2:] if _has_phase_col(X) else X.columns[1:]

    allok = True
    if isinstance(to_obs, bool):
        print('Contribution calculations need a "to_obs" argument at minimum')
        allok = False
    if cont_type == "spe" and not isinstance(from_obs, bool):
        print('For SPE contributions the "from_obs" argument is ignored')
    if (isinstance(to_obs, list) and len(to_obs) > 1) or (isinstance(from_obs, list) and len(from_obs) > 1):
        print("Contributions for groups of observations will be averaged")
    if (cont_type == "scores") and isinstance(lv_space, bool):
        print("No dimensions specified, doing contributions across all dimensions")

    if not allok:
        return None

    Xuf, _, _ = unfold_horizontal(X)
    bnames = Xuf[Xuf.columns[0]].values.tolist()
    assert not isinstance(to_obs, bool)
    to_o = [bnames.index(to_obs[i]) for i in np.arange(len(to_obs))]
    if (cont_type == "scores") or (cont_type == "ht2"):
        if not isinstance(from_obs, bool):
            from_o = [bnames.index(from_obs[i]) for i in np.arange(len(from_obs))]
        else:
            from_o = False
        cont_vec = _phi_contributions(mmvmobj, Xuf, cont_type, Y=False, from_obs=from_o, to_obs=to_o, lv_space=lv_space)  # type: ignore[arg-type]
    elif cont_type == "spe":
        cont_vec = _phi_contributions(mmvmobj, Xuf, cont_type, Y=False, from_obs=False, to_obs=to_o)  # type: ignore[arg-type]
    else:
        raise ValueError("cont_type must be 'scores' | 'ht2' | 'spe'")

    ymin = float(np.min(cont_vec))
    ymax = float(np.max(cont_vec))

    if dyn_conts:
        batch_contribs = refold_horizontal(cont_vec, mmvmobj["nvars"], mmvmobj["nsamples"])
        batch_contribs = pd.DataFrame(batch_contribs, columns=bvars)
        oids = ["Batch Contributions"] * batch_contribs.shape[0]
        batch_contribs.insert(0, "BatchID", oids)
        _plot_contribs(
            batch_contribs,
            (ymin, ymax),
            phase_samples=phase_samples,
            var_list=which_var,
            plot_title_=plot_title,
        )

    batch_contribs = refold_horizontal(np.abs(cont_vec), mmvmobj["nvars"], mmvmobj["nsamples"])
    batch_contribs = np.sum(batch_contribs, axis=0)
    plt.figure()
    plt.bar(bvars, batch_contribs)
    plt.xticks(rotation=75)
    plt.ylabel(r"$\Sigma$ (|Contribution to " + cont_type + "| )")
    plt.title(plot_title)
    plt.tight_layout()
    return cont_vec


def build_rel_time(bdata: pd.DataFrame, *, time_unit: str = "min") -> pd.DataFrame:
    """Convert 'Timestamp' to relative time per batch."""
    aux = bdata.drop_duplicates(subset=bdata.columns[0], keep="first")
    unique_batches = aux[aux.columns[0]].values.tolist()
    bdata_n: list[pd.DataFrame] = []
    for b in unique_batches:
        this_batch = bdata[bdata[bdata.columns[0]] == b].copy()
        ts = pd.to_datetime(this_batch["Timestamp"].values)
        deltat = ts - ts[0]
        tim = np.array([np.timedelta64(deltat[i], "s").astype(float) for i in np.arange(len(deltat))])
        if time_unit == "min":
            tim = tim / 60
        if time_unit == "hr":
            tim = tim / 3600
        cols = this_batch.columns.tolist()
        this_batch.insert(cols.index("Timestamp"), "Time (" + time_unit + ")", tim)
        this_batch = this_batch.drop("Timestamp", axis=1)
        bdata_n.append(this_batch)
    return pd.concat(bdata_n, axis=0)


def descriptors(bdata: pd.DataFrame, which_var: list[str], desc: list[str], *, phase: list[str] | bool = False) -> pd.DataFrame:
    """Get descriptor values for a batch trajectory."""
    has_phase = _has_phase_col(bdata)
    phase_col = bdata.columns[1] if has_phase else None
    if (not has_phase) and (not isinstance(phase, bool)):
        print("Cannot process phase flag without phase id in data")
        phase = False

    desc_val: list[list[float]] = []
    desc_lbl: list[str] = []
    for i, b in enumerate(unique(bdata, bdata.columns[0])):
        this_batch = bdata[bdata[bdata.columns[0]] == b]
        desc_val_: list[float] = []
        for v in which_var:
            if not isinstance(phase, bool):
                for p in phase:
                    values = this_batch[v][this_batch[phase_col] == p].values.astype(float)  # type: ignore[index]
                    values = values[~(np.isnan(values))]
                    for d in desc:
                        if d == "min":
                            desc_val_.append(float(values.min()))
                        if d == "max":
                            desc_val_.append(float(values.max()))
                        if d == "mean":
                            desc_val_.append(float(values.mean()))
                        if d == "median":
                            desc_val_.append(float(np.median(values)))
                        if d == "std":
                            desc_val_.append(float(np.std(values, ddof=1)))
                        if d == "var":
                            desc_val_.append(float(np.var(values, ddof=1)))
                        if d == "range":
                            desc_val_.append(float(values.max() - values.min()))
                        if d == "ave_slope":
                            x = np.arange(0, len(values))
                            m = (1 / sum(x**2)) * sum(x * (values - values[0]))
                            desc_val_.append(float(m))
                        if i == 0:
                            desc_lbl.append(v + "_" + p + "_" + d)
            else:
                values = this_batch[v].values.astype(float)
                values = values[~(np.isnan(values))]
                for d in desc:
                    if d == "min":
                        desc_val_.append(float(values.min()))
                    if d == "max":
                        desc_val_.append(float(values.max()))
                    if d == "mean":
                        desc_val_.append(float(values.mean()))
                    if d == "median":
                        desc_val_.append(float(np.median(values)))
                    if d == "std":
                        desc_val_.append(float(np.std(values, ddof=1)))
                    if d == "var":
                        desc_val_.append(float(np.var(values, ddof=1)))
                    if d == "range":
                        desc_val_.append(float(values.max() - values.min()))
                    if d == "ave_slope":
                        x = np.arange(0, len(values))
                        m = (1 / sum(x**2)) * sum(x * (values - values[0]))
                        desc_val_.append(float(m))
                    if i == 0:
                        desc_lbl.append(v + "_" + d)

        desc_val.append(desc_val_)

    bnames = unique(bdata, bdata.columns[0])
    desc_val_arr = np.array(desc_val)
    descriptors_df = pd.DataFrame(desc_val_arr, columns=desc_lbl)
    descriptors_df.insert(0, bdata.columns[0], bnames)
    return descriptors_df


def clean_empty_rows(X: pd.DataFrame, *, shush: bool = False) -> pd.DataFrame:
    """Legacy batch-oriented empty-row removal (returns DataFrame only)."""
    if _has_phase_col(X):
        X_ = np.array(X.values[:, 2:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str).tolist()
    else:
        X_ = np.array(X.values[:, 1:]).astype(float)
        ObsID_ = X.values[:, 0].astype(str).tolist()

    X_nan_map = np.isnan(X_)
    Xmiss = X_nan_map.astype(int)
    Xmiss = np.sum(Xmiss, axis=1)
    indx = [i for (i, val) in enumerate(Xmiss) if val == X_.shape[1]]

    if len(indx) > 0:
        for i in indx:
            if not shush:
                print("Removing row from ", ObsID_[i], " due to 100% missing data")
        return X.drop(X.index.values[indx].tolist())
    return X


__all__ = [
    "unique",
    "mean",
    "simple_align",
    "phase_simple_align",
    "phase_iv_align",
    "plot_var_all_batches",
    "plot_batch",
    "unfold_horizontal",
    "refold_horizontal",
    "loadings",
    "loadings_abs_integral",
    "batch_vip",
    "r2pv",
    "mpca",
    "mpls",
    "monitor",
    "predict",
    "contributions",
    "build_rel_time",
    "descriptors",
    "phase_sampling_dist",
    "clean_empty_rows",
]


