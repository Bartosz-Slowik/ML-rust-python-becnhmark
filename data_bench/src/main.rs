use std::time::Instant;
use polars::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

fn generate_dataframe(size: usize) -> DataFrame {
    let mut rng = StdRng::seed_from_u64(42);
    
    let id = Series::new("id".into(), (0..size as u32).collect::<Vec<u32>>());
    
    let mut a_values = Vec::with_capacity(size);
    let mut b_values = Vec::with_capacity(size);
    for _ in 0..size {
        a_values.push(rng.gen::<f64>());
        b_values.push(rng.gen::<f64>());
    }
    
    let mut c_values = Vec::with_capacity(size);
    for _ in 0..size {
        c_values.push(["x", "y", "z"][rng.gen_range(0..3)]);
    }
    
    let mut d_values = Vec::with_capacity(size);
    for _ in 0..size {
        d_values.push(rng.gen_range(1..100) as i32);
    }
    
    DataFrame::new(vec![
        Series::new("id".into(), id).into(),
        Series::new("a".into(), a_values).into(),
        Series::new("b".into(), b_values).into(),
        Series::new("c".into(), c_values).into(),
        Series::new("d".into(), d_values).into(),
    ]).unwrap()
}

fn benchmark_filter(df: &DataFrame) -> f64 {
    let start = Instant::now();
    
    let mask = df["a"].f64().unwrap().gt(0.5);
    let _filtered = df.filter(&mask).unwrap();
    
    start.elapsed().as_secs_f64()
}

fn benchmark_groupby(df: &DataFrame) -> f64 {
    let start = Instant::now();
    let _grouped = df.clone()
    .lazy()
    .group_by([col("c")])
    .agg([
        col("a").mean().alias("a_mean"),
        col("d").sum().alias("d_sum"),
    ])
    .collect()
    .unwrap();

    start.elapsed().as_secs_f64()
}

fn benchmark_join(df: &DataFrame, size: usize) -> f64 {
    let mut rng = StdRng::seed_from_u64(42);
    
    let id_values = (0..(size / 10) as u32).collect::<Vec<_>>();
    let mut value_values = Vec::with_capacity(size / 10);
    for _ in 0..(size / 10) {
        value_values.push(rng.gen::<f64>());
    }
    
    let df2 = DataFrame::new(vec![
        Series::new("id".into(), id_values).into(),
        Series::new("value".into(), value_values).into()
    ]).unwrap();
    
    let start = Instant::now();
    
    let _joined = df.left_join(&df2, vec!["id"], vec!["id"]).unwrap();
    
    start.elapsed().as_secs_f64()
}

fn benchmark_sort(df: &DataFrame) -> f64 {
    let start = Instant::now();
    
    let sort_options = SortOptions {
        descending: true,
        nulls_last: false,
        multithreaded: true,
        maintain_order: false,
        limit: None,
    };
    let _sorted = df.sort(vec!["a"], SortMultipleOptions::from(&sort_options)).unwrap();
    
    start.elapsed().as_secs_f64()
}

fn benchmark_calculation(df: &DataFrame) -> f64 {
    let start = Instant::now();
    
    let a = df.column("a").unwrap();
    let b = df.column("b").unwrap();
    let d = df.column("d").unwrap();
    
    let binding = ((a * b).unwrap() / d.clone()).unwrap();
    let result = binding.as_series();
    let _calc_df = df.clone().with_column(result.unwrap().clone()).unwrap();
    
    start.elapsed().as_secs_f64()
}

fn main() {
    let sizes = vec![100_000, 2_000_000, 50_000_000];
    
    println!("Rust Polars Benchmark");
    println!("{:-<70}", "");
    println!("{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}", 
             "Size", "Filter", "GroupBy", "Join", "Sort", "Calculate");
    println!("{:-<70}", "");
    
    for size in sizes {
        print!("Generating {} rows... ", size);
        let df = generate_dataframe(size);
        println!("done!");
        
        let filter_time = benchmark_filter(&df);
        let groupby_time = benchmark_groupby(&df);
        let join_time = benchmark_join(&df, size);
        let sort_time = benchmark_sort(&df);
        let calc_time = benchmark_calculation(&df);
        
        let size_str = if size >= 1_000_000 {
            format!("{:.1}M", size as f64 / 1_000_000.0)
        } else {
            format!("{}K", size / 1_000)
        };
        
        println!("{:<10} {:<12.4} {:<12.4} {:<12.4} {:<12.4} {:<12.4}", 
                size_str, filter_time, groupby_time, join_time, sort_time, calc_time);
    }
}