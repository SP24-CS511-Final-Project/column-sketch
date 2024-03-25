use column_sketch_core::NumericColumnSketch;
use column_sketch_eval::{
  column_sketch_plain_pred_greater, column_sketch_simd_pred_greater, naive_plain_pred_greater,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand_distr::{Distribution, StandardNormal};

/// Benchmark group for normally distributed f64 data.
fn bench_f64_normal(criterion: &mut Criterion) {
  // The dataset size is 50000000 f64 values, where the memory footprint is 400MB for original value
  // and 50MB for compressed codes.
  const DATA_NUM: usize = 50000000;
  let mut rng = rand::thread_rng();
  let sample: Vec<f64> = StandardNormal
    .sample_iter(&mut rng)
    .take(DATA_NUM)
    .collect();
  let sorted_sample = {
    let mut sorted = sample.clone();
    sorted.sort_by(|f1, f2| f1.partial_cmp(f2).unwrap());
    sorted
  };
  let column_sketch = NumericColumnSketch::construct_from_sorted(&sorted_sample);
  let compressed_code = column_sketch.compress_array(&sample);

  for selectivity in [0.25] {
    let index = ((sorted_sample.len() - 1) as f64 * selectivity) as usize;
    let target = sorted_sample[index];

    let group_name = format!("Benchmark f64, selectivity={}", selectivity);
    let mut group = criterion.benchmark_group(group_name);

    // Naive approach
    group.bench_with_input(
      BenchmarkId::new("Baseline", 0),
      &(&sample, target),
      |b, (sample, target)| {
        let mut output = vec![false; sample.len()];
        b.iter(|| naive_plain_pred_greater(sample, *target, &mut output));
      },
    );

    // Column sketch plain
    group.bench_with_input(
      BenchmarkId::new("Column sketch(No SIMD)", 0),
      &(&sample, &column_sketch, &compressed_code, target),
      |b, (sample, column_sketch, compressed_code, target)| {
        let mut output = vec![false; sample.len()];
        b.iter(|| {
          column_sketch_plain_pred_greater(
            sample,
            column_sketch,
            compressed_code,
            *target,
            &mut output,
          );
        })
      },
    );

    // Column sketch with simd
    group.bench_with_input(
      BenchmarkId::new("Column sketch (SIMD)", 0),
      &(&sample, &column_sketch, &compressed_code, target),
      |b, (sample, column_sketch, compressed_code, target)| {
        let mut output = vec![false; sample.len()];
        b.iter(|| {
          column_sketch_simd_pred_greater(
            sample,
            column_sketch,
            compressed_code,
            *target,
            &mut output,
          );
        })
      },
    );
  }
}

criterion_group!(benches, bench_f64_normal);
criterion_main!(benches);
