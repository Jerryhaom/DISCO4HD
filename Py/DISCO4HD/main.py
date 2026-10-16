"""DISCO scoring on the same domain and definition as the R package."""
import warnings
import numpy as np
import pandas as pd

def _matrix(frame, markers, label, minimum_rows):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError(f"{label} must be a DataFrame with unique columns")
    if not all(v in frame.columns for v in markers):
        raise ValueError(f"Biomarkers missing in {label}")
    if not all(pd.api.types.is_numeric_dtype(frame[v]) and not pd.api.types.is_bool_dtype(frame[v]) and not pd.api.types.is_complex_dtype(frame[v]) for v in markers):
        raise ValueError(f"{label} biomarkers must be real numeric")
    data = frame.loc[:, markers].to_numpy(dtype=np.float64)
    if len(data) < minimum_rows or not np.isfinite(data).all():
        raise ValueError(f"{label} must contain at least {minimum_rows} rows and finite complete biomarkers; impute explicitly")
    with np.errstate(over="ignore", invalid="ignore"):
        spread = np.std(data, axis=0, ddof=1)
    if not np.isfinite(spread).all() or np.any(spread <= 0):
        raise ValueError(f"{label} biomarkers must have positive finite variance")
    return data

def cal_disco(d4, var, ref, parallel=False, use_cpp=True, use_numba=False, ncores=4):
    """Return d4 with log DISCO; weights are estimated on all d4 rows.

    log(n_ref**2 * sum(weight * (R_ref-R_ref_plus_x)**2)) matches R.
    Zero perturbation is -Inf. Missing data are not silently imputed.
    The native backend is serial; parallel/ncores do not promise speedup.
    The historical unfinished Numba branch is explicitly unsupported.
    """
    if not isinstance(var, (list, tuple)) or len(var) < 2 or any(not isinstance(v, str) for v in var) or len(set(var)) != len(var):
        raise ValueError("var must name at least two distinct biomarkers")
    if any(not isinstance(v, (bool, np.bool_)) for v in [parallel, use_cpp, use_numba]):
        raise ValueError("Backend flags must be booleans")
    if not isinstance(ncores, (int, np.integer)) or isinstance(ncores, bool) or ncores < 1:
        raise ValueError("ncores must be a positive integer")
    if use_numba:
        raise NotImplementedError("Historical Numba code was not compiled or validated; use NumPy or C++")
    data = _matrix(d4, list(var), "d4", 2)
    reference = _matrix(ref, list(var), "ref", 3)
    if "age" not in d4 or not pd.api.types.is_numeric_dtype(d4["age"]) or pd.api.types.is_bool_dtype(d4["age"]) or pd.api.types.is_complex_dtype(d4["age"]):
        raise ValueError("A real numeric age column is required")
    age = d4["age"].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        age_spread = np.std(age, ddof=1)
    if not np.isfinite(age).all() or not np.isfinite(age_spread) or age_spread <= 0:
        raise ValueError("Age must be finite and have positive variance")
    correlations = np.array([np.corrcoef(age, data[:, j])[0, 1] for j in range(len(var))])
    weight = np.abs(np.outer(correlations, correlations)); np.fill_diagonal(weight, 0)
    total = weight.sum()
    if not np.isfinite(weight).all() or not np.isfinite(total) or total <= 0:
        raise ValueError("Age weights have zero or non-finite total")
    weight /= total
    if use_cpp:
        try:
            from .disco_optimized import disco_optimized_cpp
        except ImportError:
            warnings.warn("Optional C++ backend is not installed; using validated NumPy", RuntimeWarning, stacklevel=2)
            values = _python_disco_impl(data, reference, weight)
        else:
            values = disco_optimized_cpp(data, reference, weight).reshape(-1)
    else:
        values = _python_disco_impl(data, reference, weight)
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("DISCO calculation overflowed; check input magnitudes and units")
    return d4.assign(DISCO=values)

def _python_disco_impl(data, reference, weight):
    """Independent Pearson recomputation, not the native rank-one formula."""
    correlation = np.corrcoef(reference, rowvar=False)
    result = []
    for row in data:
        updated = np.corrcoef(np.vstack([reference, row]), rowvar=False)
        value = np.sum((correlation - updated)**2 * weight)
        with np.errstate(divide="ignore"):
            result.append(np.log(value * len(reference)**2))
    return np.asarray(result)
