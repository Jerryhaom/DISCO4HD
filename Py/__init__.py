import numpy as np
import pandas as pd
import argparse
from main import cal_disco
import os

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

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Calculate DISCO values with command line arguments')
    
    # 数据输入选项
    parser.add_argument('--input-file', '-i', type=str, 
                        help='Path to input data file (CSV format)')
    parser.add_argument('--generate-sample', action='store_true',
                        help='Generate sample data instead of reading from file')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of samples to generate (default: 1000)')
    
    # 生物标志物选项
    parser.add_argument('--biomarkers', '-b', nargs='+', type=str,
                        help='List of biomarker columns (e.g., biomarker1 biomarker2)')
    parser.add_argument('--age-column', type=str, default='age',
                        help='Name of the age column (default: age)')
    
    # DISCO计算选项
    parser.add_argument('--young-threshold', type=int, default=30,
                        help='Age threshold for defining young reference group (default: 30)')
    parser.add_argument('--use-cpp', action='store_true',
                        help='Use C++ implementation (default: False)')
    parser.add_argument('--use-numba', action='store_true',
                        help='Use Numba optimization for Python implementation (default: False)')
    
    # 输出选项
    parser.add_argument('--output-file', '-o', type=str,
                        help='Path to output file (CSV format)')
    
    return parser.parse_args()

def main():
    """主函数：处理命令行参数并计算DISCO"""
    args = parse_arguments()
    
    # 处理数据输入
    if args.generate_sample:
        print("Generating sample data with 10 biomarkers...")
        data = generate_sample_data(args.sample_size)
        print(f"Sample data generated with {args.sample_size} samples")
    elif args.input_file and os.path.exists(args.input_file):
        print(f"Reading data from {args.input_file}...")
        data = pd.read_csv(args.input_file)
        print(f"Data read with {len(data)} samples")
    else:
        print("Please provide either --generate-sample or a valid --input-file")
        return
    
    # 验证生物标志物参数
    if not args.biomarkers:
        # 默认使用10个生物标志物
        args.biomarkers = [f'biomarker{i}' for i in range(1, 11)]
        print(f"Using default biomarkers: {args.biomarkers}")
    else:
        # 检查生物标志物是否存在于数据中
        missing = [bm for bm in args.biomarkers if bm not in data.columns]
        if missing:
            print(f"Warning: Biomarkers {missing} not found in data, using available biomarkers")
            args.biomarkers = [bm for bm in args.biomarkers if bm in data.columns]
    
    # 准备参考组数据
    ref_young = data[data[args.age_column] < args.young_threshold].copy()
    if len(ref_young) == 0:
        print(f"Error: No samples found in reference group (age < {args.young_threshold})")
        return
    
    print(f"Calculating DISCO with {len(args.biomarkers)} biomarkers...")
    print(f"Reference group size: {len(ref_young)}")
    
    # 计算DISCO
    result = cal_disco(
        d4=data,
        var=args.biomarkers,
        ref=ref_young,
        use_cpp=args.use_cpp,
        use_numba=args.use_numba
    )
    print("\nDISCO Results (first 10 entries):")
    print(result.head(10))
    
    """
    result = cal_disco(
        d4=data,
        var=args.biomarkers,
        ref=ref_young,
        use_cpp=False,
        use_numba=True
    )
    """

    # 保存结果到文件
    if args.output_file:
        print(f"Saving results to {args.output_file}...")
        result.to_csv(args.output_file, index=False)
        print("Results saved successfully")

if __name__ == "__main__":
    main()
