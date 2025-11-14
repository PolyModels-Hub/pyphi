"""Utility helpers duplicated from the legacy pyphi.py module."""

from __future__ import annotations

import numpy as np


def mean(X: np.ndarray) -> np.ndarray:
    """Column-wise mean that ignores NaN values (duplicated from pyphi.py)."""
    X_nan_map = np.isnan(X)
    X_ = X.copy()
    if X_nan_map.any():
        X_nan_map = X_nan_map * 1
        X_[X_nan_map == 1] = 0
        aux = np.sum(X_nan_map, axis=0)
        x_mean = np.sum(X_, axis=0, keepdims=1) / (
            np.ones((1, X_.shape[1])) * X_.shape[0] - aux
        )
    else:
        x_mean = np.mean(X_, axis=0, keepdims=1)
    return x_mean


def std(X: np.ndarray) -> np.ndarray:
    """Column-wise standard deviation ignoring NaN values (duplicated)."""
    x_mean = mean(X)
    x_mean = np.tile(x_mean, (X.shape[0], 1))
    X_nan_map = np.isnan(X)
    if X_nan_map.any():
        X_nan_map = X_nan_map * 1
        X_ = X.copy()
        X_[X_nan_map == 1] = 0
        aux_mat = (X_ - x_mean) ** 2
        aux_mat[X_nan_map == 1] = 0
        aux = np.sum(X_nan_map, axis=0)
        x_std = np.sqrt(
            (np.sum(aux_mat, axis=0, keepdims=1))
            / (np.ones((1, X_.shape[1])) * (X_.shape[0] - 1 - aux))
        )
    else:
        x_std = np.sqrt(
            np.sum((X - x_mean) ** 2, axis=0, keepdims=1)
            / (np.ones((1, X.shape[1])) * (X.shape[0] - 1))
        )
    return x_std


def meancenterscale(X: np.ndarray, *, mcs: bool | str = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-centre and/or scale matrix (duplicated from pyphi.py)."""
    if isinstance(mcs, bool):
        if mcs:
            x_mean = mean(X)
            x_std = std(X)
            X_proc = X - np.tile(x_mean, (X.shape[0], 1))
            X_proc = X_proc / np.tile(x_std, (X.shape[0], 1))
        else:
            X_proc = X
            x_mean = np.nan
            x_std = np.nan
    elif mcs == "center":
        x_mean = mean(X)
        X_proc = X - np.tile(x_mean, (X.shape[0], 1))
        x_std = np.ones((1, X.shape[1]))
    elif mcs == "autoscale":
        x_std = std(X)
        X_proc = X / np.tile(x_std, (X.shape[0], 1))
        x_mean = np.zeros((1, X.shape[1]))
    else:
        X_proc = X
        x_mean = np.nan
        x_std = np.nan
    return X_proc, x_mean, x_std
