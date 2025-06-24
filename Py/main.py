import numpy as np
import pandas as pd
from disco_optimized import disco_optimized_cpp

def cal_disco(d4, var, ref, parallel=False, use_cpp=True, use_numba=False, ncores=4):
    """
    Calculate Distance of Covariance (DISCO)
    
    Quantifies homeostatic dysregulation by comparing biomarker covariance
    between target population and young reference population.
    
    Parameters
    ----------
    d4 : pandas.DataFrame
        Subject-level data with age and biomarkers
    var : list
        Biomarker column names for analysis
    ref : pandas.DataFrame
        Reference data from young population
    parallel : bool, optional
        Whether to use parallel computation (default: False)
    cpp : bool, optional
        Whether to use optimized implementation (default: False)
    ncores : int, optional
        Number of CPU cores for parallelization (default: 4)
    
    Returns
    -------
    pandas.DataFrame
        Original dataframe with added 'DISCO' column containing DISCO values
    
    Notes
    -----
    Algorithm:
    1. Calculate biomarker-age correlations to derive weighting matrix
    2. Compare covariance structures:
       - Target: Covariance of reference + single subject
       - Reference: Covariance of young population
    3. Output: log-transformed weighted matrix differences
    """
    # 输入验证
    if not all(v in d4.columns for v in var):
        raise ValueError("Variables missing in input data")
    if 'age' not in d4.columns:
        raise ValueError("Age column required")
    if not all(v in ref.columns for v in var):
        raise ValueError("Some variables not found in ref")

    # 计算权重矩阵
    cc = np.array([np.corrcoef(d4['age'], d4[v])[0, 1] for v in var])
    weight = np.outer(np.abs(cc), np.abs(cc))
    np.fill_diagonal(weight, 0)
    weight /= np.sum(weight)
    
    # 准备数据
    d_data = d4[var].values.astype(np.float64)
    d_ref = ref[var].values.astype(np.float64)
    
    if use_cpp:
        # 调用C++实现
        disco_values = disco_optimized_cpp(d_data, d_ref, weight)
    elif use_numba:
        disco_values = _disco_numba_optimized(
            d_data, d_ref, weight, 
            parallel=parallel, 
            ncores=ncores
        )
    else:
        disco_values = _python_disco_impl(d_data, d_ref, weight)
    
    return d4.assign(DISCO=disco_values.flatten())

#@jit(nopython=True, nogil=True, cache=True)
def _disco_numba_optimized(
    d_data: np.ndarray,
    d_ref: np.ndarray,
    weight: np.ndarray,
    parallel: bool = True,
    ncores: int = 4
) -> np.ndarray:
    """
    Numba优化核心计算
    比纯Python快10-100倍
    """
    n_ref = d_ref.shape[0]
    p = d_ref.shape[1]
    results = np.zeros(d_data.shape[0])
    ref_corr = np.corrcoef(d_ref, rowvar=False)

    # 并行计算
    for i in prange(d_data.shape[0]) if parallel else range(d_data.shape[0]):
        combined = np.vstack((d_ref, d_data[i:i+1]))
        cc1 = np.corrcoef(combined, rowvar=False)
        ds = np.nansum((ref_corr - cc1)**2 * weight)
        results[i] = np.log(ds * n_ref**2) if ds > 0 else np.log(1e-300)

    return results

def _python_disco_impl(d_data, d_ref, weight):
    """Python实现（备用）"""
    ref_corr = np.corrcoef(d_ref, rowvar=False)
    results = []
    for i in range(d_data.shape[0]):
        combined = np.vstack([d_ref, d_data[i]])
        cc1 = np.corrcoef(combined, rowvar=False)
        ds = np.nansum((ref_corr - cc1)**2 * weight)
        results.append(np.log(ds * d_ref.shape[0]**2))
    return np.array(results)

