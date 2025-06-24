import numpy as np
import pandas as pd
import argparse
import os
from .disco_optimized import disco_optimized_cpp
from .main import cal_disco

def generate_sample_data(n_samples=1000):
    """生成包含10个生物标志物的模拟数据"""
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.normal(50, 10, n_samples),
        'biomarker1': np.random.normal(0, 1, n_samples),
        'biomarker2': np.random.normal(5, 2, n_samples),
        'biomarker3': np.random.exponential(1, n_samples),
        'biomarker4': np.random.gamma(2, 3, n_samples),
        'biomarker5': np.random.weibull(1.5, n_samples) * 10,
        'biomarker6': np.random.beta(2, 5, n_samples) * 50,
        'biomarker7': np.random.lognormal(1, 0.5, n_samples),
        'biomarker8': np.random.uniform(10, 30, n_samples),
        'biomarker9': np.random.uniform(5, 20, n_samples),
        'biomarker10': np.random.chisquare(3, n_samples) * 10
    })
    return data

def example():
    data = generate_sample_data(2000)
    biomarkers = [f'biomarker{i}' for i in range(1, 11)]
    ref_young = data[data["age"] < 30].copy()
    print("Example data: ",data.head())

    result = cal_disco(
        d4=data,
        var=biomarkers,
        ref=ref_young,
        use_cpp=True,
        use_numba=False
    )
    print("DISCO results: ",result.head(10))

if __name__ == "__main__":
    example()
