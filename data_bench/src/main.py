import time
import pandas as pd
import numpy as np
import random

def generate_dataframe(size):
    random.seed(42)
    np.random.seed(42)
    
    df = pd.DataFrame({
        'id': range(size),
        'a': np.random.random(size),
        'b': np.random.random(size),
        'c': [random.choice(['x', 'y', 'z']) for _ in range(size)],
        'd': [random.randint(1, 99) for _ in range(size)]
    })
    
    return df

def benchmark_filter(df):
    start = time.time()
    
    filtered = df[df['a'] > 0.5]
    
    return time.time() - start

def benchmark_groupby(df):
    start = time.time()
    
    grouped = df.groupby('c').agg({
        'a': 'mean',
        'd': 'sum'
    }).rename(columns={'a': 'a_mean', 'd': 'd_sum'})
    
    return time.time() - start

def benchmark_join(df, size):
    np.random.seed(42)
    
    df2 = pd.DataFrame({
        'id': range(size // 10),
        'value': np.random.random(size // 10)
    })
    
    start = time.time()
    
    joined = df.merge(df2, on='id', how='left')
    
    return time.time() - start

def benchmark_sort(df):
    start = time.time()
    
    sorted_df = df.sort_values('a', ascending=False)
    
    return time.time() - start

def benchmark_calculation(df):
    start = time.time()
    
    calc_df = df.copy()
    calc_df['result'] = (df['a'] * df['b']) / df['d']
    
    return time.time() - start

def main():
    sizes = [100_000, 2_000_000, 50_000_000]
    
    print("Python Pandas Benchmark")
    print("-" * 70)
    print(f"{'Size':<10} {'Filter':<12} {'GroupBy':<12} {'Join':<12} {'Sort':<12} {'Calculate':<12}")
    print("-" * 70)
    
    for size in sizes:
        print(f"Generating {size} rows... ", end="", flush=True)
        df = generate_dataframe(size)
        print("done!")
        
        filter_time = benchmark_filter(df)
        groupby_time = benchmark_groupby(df)
        join_time = benchmark_join(df, size)
        sort_time = benchmark_sort(df)
        calc_time = benchmark_calculation(df)
        
        if size >= 1_000_000:
            size_str = f"{size/1_000_000:.1f}M"
        else:
            size_str = f"{size//1_000}K"
        
        print(f"{size_str:<10} {filter_time:<12.4f} {groupby_time:<12.4f} {join_time:<12.4f} {sort_time:<12.4f} {calc_time:<12.4f}")

if __name__ == "__main__":
    main()