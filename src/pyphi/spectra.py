"""Spectral preprocessing functions for chemometrics.

This module provides standard spectral preprocessing techniques commonly used
in near-infrared (NIR) spectroscopy and other spectroscopic applications.

All functions support both NumPy arrays and pandas DataFrames.
For DataFrames, the first column is assumed to be observation IDs and is preserved.

Author: Salvador Garcia-Munoz (sgarciam@ic.ac.uk)
Contributors: Ethan Lavialle, Carlos Perez-Galvan
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from math import factorial

__all__ = [
    'spectra_snv',
    'spectra_savgol',
    'spectra_mean_center',
    'spectra_autoscale',
    'spectra_baseline_correction',
    'spectra_msc',
]


def spectra_snv(x):
    """Apply Standard Normal Variate (SNV) transform to spectroscopic data.
    
    SNV is a row-wise normalization technique that removes scatter effects
    by centering and scaling each spectrum to zero mean and unit variance.
    
    Args:
        x: Spectra as NumPy array or pandas DataFrame
           - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
           - If DataFrame: first column is observation IDs, rest are wavelengths
    
    Returns:
        Transformed spectra in the same format as input
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100)  # 10 spectra, 100 wavelengths
        >>> spectra_normalized = spectra_snv(spectra)
        >>> np.allclose(spectra_normalized.mean(axis=1), 0, atol=1e-10)
        True
    """
    if isinstance(x, pd.DataFrame):
        x_columns = x.columns
        x_values = x.values.copy()
        x_values[:, 1:] = spectra_snv(x_values[:, 1:].astype(float))
        xpd = pd.DataFrame(x_values, columns=x_columns)
        return xpd
    else:
        if x.ndim == 2:
            # Row-wise mean and std (each spectrum separately)
            mean_x = np.mean(x, axis=1, keepdims=True)
            std_x = np.std(x, axis=1, ddof=1, keepdims=True)
            return (x - mean_x) / std_x
        else:
            # 1D case: single spectrum
            return (x - np.mean(x)) / np.std(x, ddof=1)


def spectra_savgol(ws, od, op, Dm):
    """Apply Savitzky-Golay filter for spectral smoothing and differentiation.
    
    This is a wrapper around scipy.signal.savgol_filter that maintains
    backward compatibility with the original PyPhi implementation.
    
    Args:
        ws: Window size (half-width, will use 2*ws+1 as total window)
        od: Order of derivative (0=smoothing, 1=1st derivative, 2=2nd derivative)
        op: Order of polynomial fit
        Dm: Spectra as NumPy array or pandas DataFrame
            - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
            - If DataFrame: first column is observation IDs, rest are wavelengths
    
    Returns:
        tuple: (Dm_sg, M)
            - Dm_sg: Filtered spectra (edges trimmed by window size)
            - M: Transformation matrix for applying same filter to new data
    
    Notes:
        The output will be shorter by 2*ws wavelengths due to edge trimming.
        For DataFrame input, column names are adjusted accordingly.
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100)
        >>> smoothed, transform = spectra_savgol(ws=5, od=0, op=2, Dm=spectra)
        >>> smoothed.shape
        (10, 90)  # Trimmed by 2*5 on each end
    """
    if isinstance(Dm, pd.DataFrame):
        x_columns = Dm.columns.tolist()
        FirstElement = [x_columns[0]]
        x_columns = x_columns[1:]
        FirstElement.extend(x_columns[ws:-ws])
        x_values = Dm.values
        Col1 = Dm.values[:, 0].tolist()
        Col1 = np.reshape(Col1, (-1, 1))
        aux, M = spectra_savgol(ws, od, op, x_values[:, 1:].astype(float))
        data_ = np.hstack((Col1, aux))
        xpd = pd.DataFrame(data=data_, columns=FirstElement)
        return xpd, M
    else:
        if Dm.ndim == 1:
            l = Dm.shape[0]
        else:
            l = Dm.shape[1]
        
        # Build Savitzky-Golay coefficients matrix
        x_vec = np.arange(-ws, ws + 1)
        x_vec = np.reshape(x_vec, (len(x_vec), 1))
        X = np.ones((2 * ws + 1, 1))
        for oo in np.arange(1, op + 1):
            X = np.hstack((X, x_vec ** oo))
        
        try:
            XtXiXt = np.linalg.inv(X.T @ X) @ X.T
        except:
            XtXiXt = np.linalg.pinv(X.T @ X) @ X.T
        
        coeffs = XtXiXt[od, :] * factorial(od)
        coeffs = np.reshape(coeffs, (1, len(coeffs)))
        
        # Build transformation matrix M
        for i in np.arange(1, l - 2 * ws + 1):
            if i == 1:
                M = np.hstack((coeffs, np.zeros((1, l - 2 * ws - 1))))
            elif i < l - 2 * ws:
                m_ = np.hstack((np.zeros((1, i - 1)), coeffs))
                m_ = np.hstack((m_, np.zeros((1, l - 2 * ws - 1 - i + 1))))
                M = np.vstack((M, m_))
            else:
                m_ = np.hstack((np.zeros((1, l - 2 * ws - 1)), coeffs))
                M = np.vstack((M, m_))
        
        # Apply transformation
        if Dm.ndim == 1:
            Dm_sg = M @ Dm
        else:
            Dm_sg = Dm @ M.T
        
        return Dm_sg, M


def spectra_mean_center(Dm):
    """Mean center each spectrum to have zero mean.
    
    Subtracts the mean of each spectrum from itself, removing baseline offset.
    This is a row-wise operation (across wavelengths for each spectrum).
    
    Args:
        Dm: Spectra as NumPy array or pandas DataFrame
            - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
            - If DataFrame: first column is observation IDs, rest are wavelengths
    
    Returns:
        Mean-centered spectra in the same format as input
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100) + 5  # Add offset
        >>> centered = spectra_mean_center(spectra)
        >>> np.allclose(centered.mean(axis=1), 0, atol=1e-10)
        True
    """
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values.copy()
        Dm_values[:, 1:] = spectra_mean_center(Dm_values[:, 1:].astype(float))
        Dm_pd = pd.DataFrame(Dm_values, columns=Dm_columns)
        return Dm_pd
    else:
        if Dm.ndim == 2:
            spectra_mean = Dm.mean(axis=1, keepdims=True)
            return Dm - spectra_mean
        elif Dm.ndim == 1:
            return Dm - Dm.mean()
        else:
            raise ValueError(f"Expected 1D or 2D array, got {Dm.ndim}D")


def spectra_autoscale(Dm):
    """Autoscale each spectrum to have unit variance.
    
    Divides each spectrum by its standard deviation, normalizing the scale.
    This is a row-wise operation (across wavelengths for each spectrum).
    
    Args:
        Dm: Spectra as NumPy array or pandas DataFrame
            - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
            - If DataFrame: first column is observation IDs, rest are wavelengths
    
    Returns:
        Autoscaled spectra in the same format as input
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100) * 5  # Different scales
        >>> scaled = spectra_autoscale(spectra)
        >>> np.allclose(scaled.std(axis=1, ddof=1), 1.0, atol=1e-10)
        True
    """
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values.copy()
        Dm_values[:, 1:] = spectra_autoscale(Dm_values[:, 1:].astype(float))
        Dm_pd = pd.DataFrame(Dm_values, columns=Dm_columns)
        return Dm_pd
    else:
        if Dm.ndim == 2:
            # ddof=1 for sample standard deviation (N-1 denominator)
            spectra_sd = Dm.std(axis=1, ddof=1, keepdims=True)
            return Dm / spectra_sd
        elif Dm.ndim == 1:
            return Dm / Dm.std(ddof=1)
        else:
            raise ValueError(f"Expected 1D or 2D array, got {Dm.ndim}D")


def spectra_baseline_correction(Dm):
    """Shift each spectrum to have minimum value of zero.
    
    Subtracts the minimum value from each spectrum, removing baseline offset.
    This is a row-wise operation (across wavelengths for each spectrum).
    
    Args:
        Dm: Spectra as NumPy array or pandas DataFrame
            - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
            - If DataFrame: first column is observation IDs, rest are wavelengths
    
    Returns:
        Baseline-corrected spectra in the same format as input
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100) + 10  # Positive offset
        >>> corrected = spectra_baseline_correction(spectra)
        >>> np.allclose(corrected.min(axis=1), 0, atol=1e-10)
        True
    """
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values.copy()
        Dm_values[:, 1:] = spectra_baseline_correction(Dm_values[:, 1:].astype(float))
        Dm_pd = pd.DataFrame(Dm_values, columns=Dm_columns)
        return Dm_pd
    else:
        if Dm.ndim == 2:
            spectra_min = Dm.min(axis=1, keepdims=True)
            return Dm - spectra_min
        elif Dm.ndim == 1:
            return Dm - Dm.min()
        else:
            raise ValueError(f"Expected 1D or 2D array, got {Dm.ndim}D")


def spectra_msc(Dm, reference_spectra=None):
    """Apply Multiplicative Scatter Correction (MSC) to spectra.
    
    MSC removes multiplicative scatter effects by fitting each spectrum
    to a reference spectrum using a linear model: spectrum = a + b * reference.
    The corrected spectrum is then (spectrum - a) / b.
    
    Args:
        Dm: Spectra as NumPy array or pandas DataFrame
            - If array: shape (n_samples, n_wavelengths) or (n_wavelengths,)
            - If DataFrame: first column is observation IDs, rest are wavelengths
        reference_spectra: Optional reference spectrum (1D array)
            - If None, uses the mean of all spectra (only valid for 2D input)
            - For 1D input, reference_spectra must be provided
    
    Returns:
        MSC-corrected spectra in the same format as input
    
    Raises:
        ValueError: If reference_spectra is None for 1D input
    
    Examples:
        >>> import numpy as np
        >>> spectra = np.random.randn(10, 100)
        >>> corrected = spectra_msc(spectra)  # Uses mean as reference
        >>> corrected.shape
        (10, 100)
    """
    if isinstance(Dm, pd.DataFrame):
        Dm_columns = Dm.columns
        Dm_values = Dm.values.copy()
        Dm_values[:, 1:] = spectra_msc(Dm_values[:, 1:].astype(float), reference_spectra)
        Dm_pd = pd.DataFrame(Dm_values, columns=Dm_columns)
        return Dm_pd
    else:
        if Dm.ndim == 2:
            # Use mean spectrum as reference if not provided
            if reference_spectra is None:
                reference_spectra = Dm.mean(axis=0)
            
            # Build design matrix [ones, reference]
            V = np.vstack([np.ones(reference_spectra.shape), reference_spectra])
            
            # Solve for coefficients: Dm = U @ V where U = [a, b] for each spectrum
            U = Dm @ V.T @ np.linalg.inv(V @ V.T)
            
            # Correct: (spectrum - a) / b
            corrected_spectra = (Dm - U[:, 0, None]) / U[:, 1, None]
            return corrected_spectra
        else:
            # 1D case: single spectrum
            if reference_spectra is None:
                raise ValueError(
                    "MSC requires a reference spectrum for 1D input. "
                    "Provide reference_spectra or use 2D input to compute mean."
                )
            
            # Build design matrix [ones, reference]
            V = np.vstack([np.ones(reference_spectra.shape), reference_spectra])
            
            # Solve for coefficients
            U = Dm @ V.T @ np.linalg.inv(V @ V.T)
            
            # Correct: (spectrum - a) / b
            corrected_spectra = (Dm - U[0]) / U[1]
            return corrected_spectra
