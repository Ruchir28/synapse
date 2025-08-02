use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use synapse::{NDArray, ops::dot::DotOps};

fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("Dot");

    let shape_a = vec![1024, 1024];
    let size_a: usize = shape_a.iter().product();
    let a = NDArray::new((0..size_a).map(|x| x as f32).collect(), shape_a.clone());

    let shape_b = vec![1024, 1024];
    let size_b: usize = shape_b.iter().product();
    let b = NDArray::new((0..size_b).map(|x| x as f32).collect(), shape_b.clone());

    group.bench_function(BenchmarkId::new("SIMD", size_a), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).dot(black_box(&b));
        })
    });

    group.bench_function(BenchmarkId::new("Scalar", size_a), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).fallback_dot(black_box(&b));
        })
    });

    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add");

    let shape = vec![1024, 1024];
    let size: usize = shape.iter().product();
    let a = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());
    let b = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());

    // Benchmark the main `try_add` function (which should use SIMD)
    group.bench_function(BenchmarkId::new("SIMD", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).try_add(black_box(&b));
        })
    });

    // Benchmark the pure scalar `fallback_add` function
    group.bench_function(BenchmarkId::new("Scalar", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).fallback_add(black_box(&b));
        })
    });

    group.finish();
}

fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("Mul");

    let shape = vec![1024, 1024];
    let size: usize = shape.iter().product();
    let a = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());
    let b = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());

    // Benchmark the main `try_mul` function (which should use SIMD)
    group.bench_function(BenchmarkId::new("SIMD", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).try_mul(black_box(&b));
        })
    });

    // Benchmark the pure scalar `fallback_mul` function
    group.bench_function(BenchmarkId::new("Scalar", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).fallback_mul(black_box(&b));
        })
    });

    group.finish();
}

fn bench_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("Sub");

    let shape = vec![1024, 1024];
    let size: usize = shape.iter().product();
    let a = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());
    let b = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());

    // Benchmark the main `try_sub` function (which should use SIMD)
    group.bench_function(BenchmarkId::new("SIMD", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).try_sub(black_box(&b));
        })
    });

    // Benchmark the pure scalar `fallback_sub` function
    group.bench_function(BenchmarkId::new("Scalar", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).fallback_sub(black_box(&b));
        })
    });

    group.finish();
}

fn bench_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("Div");

    let shape = vec![1024, 1024];
    let size: usize = shape.iter().product();
    let a = NDArray::new((0..size).map(|x| x as f32).collect(), shape.clone());
    let b = NDArray::new((1..=size).map(|x| x as f32).collect(), shape.clone()); // Avoid division by zero

    // Benchmark the main `try_div` function (which should use SIMD)
    group.bench_function(BenchmarkId::new("SIMD", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).try_div(black_box(&b));
        })
    });

    // Benchmark the pure scalar `fallback_div` function
    group.bench_function(BenchmarkId::new("Scalar", size), |bencher| {
        bencher.iter(|| {
            let _result = black_box(&a).fallback_div(black_box(&b));
        })
    });

    group.finish();
}

criterion_group!(benches, bench_add, bench_mul, bench_sub, bench_div, bench_dot);
criterion_main!(benches);
