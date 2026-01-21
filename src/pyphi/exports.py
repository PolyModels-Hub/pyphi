"""Model export and format conversion functions.

This module provides functions to export PyPhi models to various external formats
and convert model structures for use in optimization and simulation software.

Common use cases:
- Export PLS models to gPROMS for hybrid modeling
- Adapt PLS models for Pyomo optimization
- Parse materials data from Excel for JRPLS modeling
- Convert categorical data to binary matrices

Author: Salvador Garcia-Munoz (sgarciam@ic.ac.uk)
Contributors: Ethan Lavialle, Carlos Perez-Galvan
"""

import numpy as np
import pandas as pd
from ._internal import np2D2pyomo, np1D2pyomo
from .utils import unique

__all__ = [
    'export_2_gproms',
    'adapt_pls_4_pyomo',
    'conv_pls_2_eiot',
    'cat_2_matrix',
    'parse_materials',
]


def export_2_gproms(mvmobj, *, fname='phi_export.txt'):
    """Export PLS model to gPROMS format for hybrid modeling.
    
    This function exports a PLS model in gPROMS syntax, allowing the model
    to be embedded in process simulation and optimization workflows.
    
    The exported file defines all PLS model parameters (means, scales, weights,
    loadings, etc.) and equations needed to use the model in gPROMS.
    
    Args:
        mvmobj: A PLS model created with pyphi.pls
        fname: Name of the text file to be created (default: 'phi_export.txt')
    
    Returns:
        None (writes file to disk)
    
    Notes:
        Typical usage in gPROMS involves using variables X_NEW (input) and 
        Y_PRED (output) as the interface to the PLS model.
        
        The exported model includes:
        - Parameter declarations (X_VARS, Y_VARS, A)
        - Variable declarations (means, stds, weights, loadings)
        - Equations for prediction, reconstruction, T², SPE
    
    Examples:
        >>> plsobj = phi.pls(X, Y, 3)
        >>> phi.export_2_gproms(plsobj, fname='my_model.txt')
        # Creates my_model.txt with gPROMS-compatible model definition
    """
    top_lines = [
        'PARAMETER',
        'X_VARS AS ORDERED_SET',
        'Y_VARS AS ORDERED_SET',
        'A      AS INTEGER',
        'VARIABLE',
        'X_MEANS as ARRAY(X_VARS)    OF no_type',
        'X_STD   AS ARRAY(X_VARS)    OF no_type',
        'Y_MEANS as ARRAY(Y_VARS)    OF no_type',
        'Y_STD   AS ARRAY(Y_VARS)    OF no_type',
        'Ws      AS ARRAY(X_VARS,A)  OF no_type',
        'Q       AS ARRAY(Y_VARS,A)  OF no_type',
        'P       AS ARRAY(X_VARS,A)  OF no_type',
        'T       AS ARRAY(A)         OF no_type',
        'Tvar    AS ARRAY(A)         OF no_type',
        'X_HAT   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
        'Y_HAT   AS ARRAY(Y_VARS)    OF no_type # Mean-centered and scaled',
        'X_PRED  AS ARRAY(X_VARS)    OF no_type # In original units',
        'Y_PRED  AS ARRAY(Y_VARS)    OF no_type # In original units',
        'X_NEW   AS ARRAY(X_VARS)    OF no_type # In original units',
        'X_MCS   AS ARRAY(X_VARS)    OF no_type # Mean-centered and scaled',
        'HT2                         AS no_type',
        'SPEX                        AS no_type',
        'SET'
    ]
    
    x_var_line = "X_VARS:=['" + mvmobj['varidX'][0] + "'"
    for v in mvmobj['varidX'][1:]:
        x_var_line = x_var_line + ",'" + v + "'"
    x_var_line = x_var_line + '];'

    y_var_line = "Y_VARS:=['" + mvmobj['varidY'][0] + "'"
    if len(mvmobj['varidY']) > 1:
        for v in mvmobj['varidY'][1:]:
            y_var_line = y_var_line + ",'" + v + "'"
    y_var_line = y_var_line + '];'

    top_lines.append(x_var_line)
    top_lines.append(y_var_line)
    top_lines.append('A:=' + str(mvmobj['T'].shape[1]) + ';')
    
    mid_lines = [
        'EQUATION',
        'X_MCS * X_STD = (X_NEW-X_MEANS);',
        'FOR j:=1 TO A DO',
        'T(j) = SIGMA(X_MCS*Ws(,j));',
        'END',
        'FOR i IN Y_VARS DO',
        'Y_HAT(i) = SIGMA(T*Q(i,));',
        'END',
        'FOR i IN X_VARS DO',
        'X_HAT(i) = SIGMA(T*P(i,));',
        'END',
        '(X_HAT * X_STD) + X_MEANS = X_PRED;',
        '(Y_HAT * Y_STD) + Y_MEANS = Y_PRED;',
        'HT2  = SIGMA ((T^2)/Tvar);',
        'SPEX = SIGMA ((X_MCS - X_HAT)^2);'
    ]

    assign_lines = ['ASSIGN']
    for i, xvar in enumerate(mvmobj['varidX']):
        assign_lines.append("X_MEANS('" + xvar + "') := " + str(mvmobj['mx'][0, i]) + ";")

    for i, xvar in enumerate(mvmobj['varidX']):
        assign_lines.append("X_STD('" + xvar + "') := " + str(mvmobj['sx'][0, i]) + ";")

    for i, yvar in enumerate(mvmobj['varidY']):
        assign_lines.append("Y_MEANS('" + yvar + "') := " + str(mvmobj['my'][0, i]) + ";")

    for i, yvar in enumerate(mvmobj['varidY']):
        assign_lines.append("Y_STD('" + yvar + "') := " + str(mvmobj['sy'][0, i]) + ";")

    for i, xvar in enumerate(mvmobj['varidX']):
        for j in np.arange(mvmobj['Ws'].shape[1]):
            assign_lines.append("Ws('" + xvar + "'," + str(j + 1) + ") := " + str(mvmobj['Ws'][i, j]) + ";")

    for i, xvar in enumerate(mvmobj['varidX']):
        for j in np.arange(mvmobj['P'].shape[1]):
            assign_lines.append("P('" + xvar + "'," + str(j + 1) + ") := " + str(mvmobj['P'][i, j]) + ";")
    
    for i, yvar in enumerate(mvmobj['varidY']):
        for j in np.arange(mvmobj['Q'].shape[1]):
            assign_lines.append("Q('" + yvar + "'," + str(j + 1) + ") := " + str(mvmobj['Q'][i, j]) + ";")
    
    tvar = np.std(mvmobj['T'], axis=0, ddof=1)
    for j in np.arange(mvmobj['T'].shape[1]):
        assign_lines.append("Tvar(" + str(j + 1) + ") := " + str(tvar[j]) + ";")

    lines = top_lines
    lines.extend(mid_lines)
    lines.extend(assign_lines)

    with open(fname, "w") as outfile:
        outfile.write("\n".join(lines))

    return


def adapt_pls_4_pyomo(plsobj, *, use_var_ids=False):
    """Adapt PLS model parameters for use in Pyomo optimization.
    
    Converts all NumPy arrays in a PLS model to dictionary format required by Pyomo.
    All converted parameters are added to a copy of the model with prefix 'pyo_'.
    
    Args:
        plsobj: A PLS object created with pyphi.pls
        use_var_ids: If True, use variable IDs from plsobj as dictionary keys
                     If False (default), use integer indices
    
    Returns:
        plsobj_pyomo: Dictionary with original PLS data plus Pyomo-formatted parameters
        
        Added fields:
        - pyo_A: List of latent variable indices
        - pyo_N: List of X variable indices (integers or variable IDs)
        - pyo_M: List of Y variable indices (integers or variable IDs)
        - pyo_Ws: Weights matrix in dictionary format
        - pyo_Q: Y-loadings matrix in dictionary format
        - pyo_P: X-loadings matrix in dictionary format
        - pyo_var_t: Score variances in dictionary format
        - pyo_mx, pyo_sx: X means and scales in dictionary format
        - pyo_my, pyo_sy: Y means and scales in dictionary format
        - speX_lim95: SPE limit (95% confidence)
    
    Examples:
        >>> plsobj = phi.pls(X, Y, 3)
        >>> plsobj_pyo = phi.adapt_pls_4_pyomo(plsobj, use_var_ids=True)
        >>> # Now use plsobj_pyo['pyo_Ws'], etc. in Pyomo model
    """
    plsobj_ = plsobj.copy()
    
    A = plsobj['T'].shape[1]
    N = plsobj['P'].shape[0]
    M = plsobj['Q'].shape[0]
    
    pyo_A = np.arange(1, A + 1).tolist()  # index for LV's
    
    if not use_var_ids:
        pyo_N = np.arange(1, N + 1).tolist()  # index for columns of X
        pyo_M = np.arange(1, M + 1).tolist()  # index for columns of Y
        pyo_Ws = np2D2pyomo(plsobj['Ws'])
        pyo_Q = np2D2pyomo(plsobj['Q'])
        pyo_P = np2D2pyomo(plsobj['P'])
        var_t = np.var(plsobj['T'], axis=0)
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx = np1D2pyomo(plsobj['mx'])
        pyo_sx = np1D2pyomo(plsobj['sx'])
        pyo_my = np1D2pyomo(plsobj['my'])
        pyo_sy = np1D2pyomo(plsobj['sy'])
    else:
        pyo_N = plsobj['varidX']
        pyo_M = plsobj['varidY']
        pyo_Ws = np2D2pyomo(plsobj['Ws'], varids=plsobj['varidX'])
        pyo_Q = np2D2pyomo(plsobj['Q'], varids=plsobj['varidY'])
        pyo_P = np2D2pyomo(plsobj['P'], varids=plsobj['varidX'])
        var_t = np.var(plsobj['T'], axis=0)
        pyo_var_t = np1D2pyomo(var_t)
        pyo_mx = np1D2pyomo(plsobj['mx'], indexes=plsobj['varidX'])
        pyo_sx = np1D2pyomo(plsobj['sx'], indexes=plsobj['varidX'])
        pyo_my = np1D2pyomo(plsobj['my'], indexes=plsobj['varidY'])
        pyo_sy = np1D2pyomo(plsobj['sy'], indexes=plsobj['varidY'])
    
    plsobj_['pyo_A'] = pyo_A
    plsobj_['pyo_N'] = pyo_N
    plsobj_['pyo_M'] = pyo_M
    plsobj_['pyo_Ws'] = pyo_Ws
    plsobj_['pyo_Q'] = pyo_Q
    plsobj_['pyo_P'] = pyo_P
    plsobj_['pyo_var_t'] = pyo_var_t
    plsobj_['pyo_mx'] = pyo_mx
    plsobj_['pyo_sx'] = pyo_sx
    plsobj_['pyo_my'] = pyo_my
    plsobj_['pyo_sy'] = pyo_sy
    plsobj_['speX_lim95'] = plsobj['speX_lim95']
    
    return plsobj_


def conv_pls_2_eiot(plsobj, *, r_length=False):
    """Convert PLS model to EIOT (Economic Input-Output Table) format.
    
    This is a specialized conversion for economic modeling applications where
    PLS models are embedded in input-output economic frameworks.
    
    Args:
        plsobj: A PLS object created with pyphi.pls
        r_length: Length of the R vector (resource variables)
                  - False (default): use all X variables
                  - Integer: specify number of R variables (rest are equality constraints)
    
    Returns:
        plsobj_: Dictionary with PLS model adapted for EIOT format
        
        Additional fields beyond adapt_pls_4_pyomo:
        - indx_r: Indices for R (resource) variables
        - indx_rk_eq: Indices for variables treated as equality constraints
        - var_t: Score variances (NumPy array)
        - S_I: Placeholder for sensitivity index (np.nan)
        - pyo_S_I: Placeholder for Pyomo-formatted sensitivity (np.nan)
    
    Notes:
        This function partitions X variables into resources (R) and 
        equality-constrained variables based on r_length.
    
    Examples:
        >>> plsobj = phi.pls(X, Y, 3)
        >>> plsobj_eiot = phi.conv_pls_2_eiot(plsobj, r_length=5)
        # First 5 X variables treated as resources, rest as constraints
    """
    plsobj_ = plsobj.copy()
    
    A = plsobj['T'].shape[1]
    N = plsobj['P'].shape[0]
    M = plsobj['Q'].shape[0]
    
    pyo_A = np.arange(1, A + 1).tolist()  # index for LV's
    pyo_N = np.arange(1, N + 1).tolist()  # index for columns of X
    pyo_M = np.arange(1, M + 1).tolist()  # index for columns of Y
    
    pyo_Ws = np2D2pyomo(plsobj['Ws'])
    pyo_Q = np2D2pyomo(plsobj['Q'])
    pyo_P = np2D2pyomo(plsobj['P'])
    
    var_t = np.var(plsobj['T'], axis=0)
    
    pyo_var_t = np1D2pyomo(var_t)
    pyo_mx = np1D2pyomo(plsobj['mx'])
    pyo_sx = np1D2pyomo(plsobj['sx'])
    pyo_my = np1D2pyomo(plsobj['my'])
    pyo_sy = np1D2pyomo(plsobj['sy'])
    
    # Determine R and equality constraint indices
    if not isinstance(r_length, bool):
        if r_length < N:
            indx_r = np.arange(1, r_length + 1).tolist()
            indx_rk_eq = np.arange(r_length + 1, N + 1).tolist()
        elif r_length == N:
            indx_r = pyo_N
            indx_rk_eq = 0
        else:
            print('r_length >> N !!')
            print('Forcing r_length=N')
            indx_r = pyo_N
            indx_rk_eq = 0
    else:
        if not r_length:
            indx_r = pyo_N
            indx_rk_eq = 0
    
    plsobj_['pyo_A'] = pyo_A
    plsobj_['pyo_N'] = pyo_N
    plsobj_['pyo_M'] = pyo_M
    plsobj_['pyo_Ws'] = pyo_Ws
    plsobj_['pyo_Q'] = pyo_Q
    plsobj_['pyo_P'] = pyo_P
    plsobj_['pyo_var_t'] = pyo_var_t
    plsobj_['indx_r'] = indx_r
    plsobj_['indx_rk_eq'] = indx_rk_eq
    plsobj_['pyo_mx'] = pyo_mx
    plsobj_['pyo_sx'] = pyo_sx
    plsobj_['pyo_my'] = pyo_my
    plsobj_['pyo_sy'] = pyo_sy
    plsobj_['S_I'] = np.nan
    plsobj_['pyo_S_I'] = np.nan
    plsobj_['var_t'] = var_t
    
    return plsobj_


def cat_2_matrix(X):
    """Convert categorical data to binary matrices for regression.
    
    Transforms a DataFrame with categorical variables into binary-coded matrices
    suitable for regression modeling. Useful for incorporating categorical features
    into PLS/PCA models.
    
    Args:
        X: Pandas DataFrame with categorical descriptors for each observation
           First column should be observation IDs
           Subsequent columns contain categorical variables
    
    Returns:
        tuple: (Xmat, XmatMB)
            - Xmat: DataFrame with all categories as binary columns
            - XmatMB: Dictionary with block structure for multi-block modeling
                      {'data': list of DataFrames (one per original column),
                       'blknames': list of original column names}
    
    Examples:
        >>> df = pd.DataFrame({
        ...     'ID': ['Obs1', 'Obs2', 'Obs3'],
        ...     'Color': ['Red', 'Blue', 'Red'],
        ...     'Size': ['Large', 'Small', 'Large']
        ... })
        >>> Xmat, XmatMB = phi.cat_2_matrix(df)
        >>> # Xmat has columns: ID, Red, Blue, Large, Small
        >>> # XmatMB['data'] has 2 DataFrames (one for Color, one for Size)
    
    Notes:
        - Each category becomes a binary (0/1) column
        - Original order of categories is preserved
        - Multi-block structure useful for multi-block PLS (mbpls)
    """
    FirstOne = True
    Xmat = []
    Xcat = []
    XcatMB = []
    XmatMB = []
    blknames = []
    
    for x in X:
        if not FirstOne:
            blknames.append(x)
            categories = np.unique(X[x])
            XcatMB.append(categories)
            Xmat_ = []
            for c in categories:
                Xcat.append(c)
                xmat_ = (X[x] == c) * 1
                Xmat.append(xmat_)
                Xmat_.append(xmat_)
            
            Xmat_ = np.array(Xmat_).T
            Xmat_ = pd.DataFrame(Xmat_, columns=categories)
            Xmat_.insert(0, firstcol, X[firstcol])
            XmatMB.append(Xmat_)
        else:
            firstcol = x
            FirstOne = False
    
    Xmat = np.array(Xmat).T
    Xmat = pd.DataFrame(Xmat, columns=Xcat)
    Xmat.insert(0, firstcol, X[firstcol])
    XmatMB = {'data': XmatMB, 'blknames': blknames}
    
    return Xmat, XmatMB


def parse_materials(filename, sheetname):
    """Parse materials composition data from Excel for JRPLS modeling.
    
    Reads an Excel file with materials usage data and constructs R matrices
    (composition/ratio matrices) suitable for Joint Range PLS (JRPLS) modeling.
    
    The Excel file must have four columns:
    - 'Finished Product Lot': Batch identifier
    - 'Material Lot': Specific lot of material used
    - 'Ratio or Quantity': Amount or ratio of material
    - 'Material': Type/name of material
    
    Args:
        filename: Path to Excel workbook containing the data
        sheetname: Name of the sheet in the workbook with the data
    
    Returns:
        tuple: (JR, materials_used)
            - JR: List of DataFrames, one per material type, showing usage per batch
                  Each DataFrame has finished product lots as rows, material lots as columns
            - materials_used: List of material names (order matches JR list)
            
        If data validation fails, returns (False, False)
    
    Notes:
        - Function validates that all batches have complete material information
        - Prints ratio/quantity sum for each batch (should typically sum to 1.0 for ratios)
        - For JRPLS modeling, use JR as the Ri input
    
    Examples:
        >>> JR, materials = phi.parse_materials('batch_data.xlsx', 'Materials')
        >>> # JR[0] contains ratios for first material type
        >>> # JR[1] contains ratios for second material type, etc.
        >>> jrpls_model = phi.jrpls(Xi, JR, Y, A=3)
    
    Raises:
        Prints error messages if:
        - Any batch is missing material lot information
        - Data validation fails
    """
    materials = pd.read_excel(filename, sheet_name=sheetname)

    ok = True
    for lot in unique(materials, 'Finished Product Lot'):
        this_lot = materials[materials["Finished Product Lot"] == lot]
        for mt, m in zip(this_lot['Material'].values, this_lot['Material Lot'].values):
            try:
                if np.isnan(m):
                    print('Lot ' + lot + ' has no Material Lot for ' + mt)
                    ok = False
                    break
            except:
                d = 1
        if not ok:
            break
        print('Lot :' + lot + ' ratio/qty adds to ' + str(np.sum(this_lot['Ratio or Quantity'].values)))
    
    if ok:
        JR = []
        materials_used = unique(materials, 'Material')
        fp_lots = unique(materials, 'Finished Product Lot')
        for m in materials_used:
            r_mat = []
            mat_lots = np.unique(materials['Material Lot'][materials['Material'] == m]).tolist()
            for lot in fp_lots:
                rvec = np.zeros(len(mat_lots))
                this_lot_this_mat = materials[
                    (materials["Finished Product Lot"] == lot) &
                    (materials['Material'] == m)
                ]
                for l, r in zip(this_lot_this_mat['Material Lot'].values,
                                this_lot_this_mat['Ratio or Quantity'].values):
                    rvec[mat_lots.index(l)] = r
                r_mat.append(rvec)
            r_mat_pd = pd.DataFrame(np.array(r_mat), columns=mat_lots)
            r_mat_pd.insert(0, 'FPLot', fp_lots)
            JR.append(r_mat_pd)
        return JR, materials_used
    else:
        print('Data needs revision')
        return False, False
