"""
PyPhi - Multivariate Analysis Toolbox

A comprehensive Python library for chemometrics, multivariate analysis,
and model-based diagnostics.

Version: 0.1.0
"""

from . import utils
from . import _internal
from . import pca as pca_module
from .utils import (
    # Statistical functions
    mean,
    std,
    meancenterscale,
    f95,
    f99,
    spe_ci,
    single_score_conf_int,
    # NaN utilities
    n2z,
    z2n,
    # Array utilities
    find,
    unique,
    # Data cleaning
    clean_htmls,
    clean_empty_rows,
    clean_low_variances,
    # DataFrame reconciliation
    isin_ordered_col0,
    reconcile_rows,
    reconcile_rows_to_columns,
)

# Internal functions (semi-public for advanced users)
from ._internal import (
    scores_conf_int_calc,
    np2D2pyomo,
    np1D2pyomo,
)

# PCA functions
from .pca import (
    pca,
    pca_,
    pca_pred,
    hott2,
    prep_pca_4_MDbyNLP,
)

# PLS functions
from .pls import (
    pls,
    pls_,
    pls_pred,
    spe,
    prep_pls_4_MDbyNLP,
    cca,
    cca_multi,
)

# Advanced PLS functions
from .advanced_pls import (
    lwpls,
    mbpls,
    lpls,
    lpls_pred,
    jrpls,
    jrpls_pred,
    tpls,
    tpls_pred,
)

# Diagnostics and analysis tools
from .diagnostics import (
    contributions,
    varimax_,
    varimax_rotation,
    bootstrap_pls,
    bootstrap_pls_pred,
    build_polynomial,
)

from . import batch

__version__ = "0.1.0"
__author__ = "Ethan Lavialle <ethan.lavialle@polymodelshub.com>"

__all__ = [
    "utils",
    "_internal",
    "pca_module",
    # Statistical functions
    "mean",
    "std",
    "meancenterscale",
    "f95",
    "f99",
    "spe_ci",
    "single_score_conf_int",
    # NaN utilities
    "n2z",
    "z2n",
    # Array utilities
    "find",
    "unique",
    # Data cleaning
    "clean_htmls",
    "clean_empty_rows",
    "clean_low_variances",
    # DataFrame reconciliation
    "isin_ordered_col0",
    "reconcile_rows",
    "reconcile_rows_to_columns",
    # Internal helpers (semi-public)
    "scores_conf_int_calc",
    "np2D2pyomo",
    "np1D2pyomo",
    # PCA
    "pca",
    "pca_",
    "pca_pred",
    "hott2",
    "prep_pca_4_MDbyNLP",
    # PLS
    "pls",
    "pls_",
    "pls_pred",
    "spe",
    "prep_pls_4_MDbyNLP",
    "cca",
    "cca_multi",
    # Advanced PLS
    "lwpls",
    "mbpls",
    "lpls",
    "lpls_pred",
    "jrpls",
    "jrpls_pred",
    "tpls",
    "tpls_pred",
    # Diagnostics
    "contributions",
    "varimax_",
    "varimax_rotation",
    "bootstrap_pls",
    "bootstrap_pls_pred",
    "build_polynomial",
    "batch",
]
