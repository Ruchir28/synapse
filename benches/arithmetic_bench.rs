use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use synapse::NDArray;

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

criterion_group!(benches, bench_add);
criterion_main!(benches);
