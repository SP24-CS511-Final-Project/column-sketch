//! The parquet benchmark experiment is based on the physical execution design of [arrow-datafusion](https://github.com/apache/arrow-datafusion) to simulate a realistic workflow
//! Arrow-datafusion passes `Vec<RecordBatch>` between operators, which means filter operator materializes the result rather than passing the selection list or bitvector
//! along the execution. The experiment implements the same interface.
//! The core experiment interface is a function:
//! `fn execute(path: impl Path, target_value: f64) -> Pin<Box<dyn Stream<Item=Result<RecordBatch>> + Send>>`, which accepts a parquet file and a target predicate value
//! and returns an async stream of `RecordBatch`s, which roughly matches the [`execute` API from datafusion](https://github.com/apache/arrow-datafusion/blob/1e4ddb6d86328cb6596bb50da9ccc654f19a83ea/datafusion/physical-plan/src/lib.rs#L378C5-L382)
//! The benchmark harness will create and drain the stream as a complete experiment run.

use column_sketch_eval::parquet::{
  build_column_sketch, execute_column_sketch, execute_plain_parquet,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

/// Benchmark group for normally distributed f64 data.
fn bench_parquet(criterion: &mut Criterion) {
  let files = [
    "../pqs/0.2.parquet",
    "../pqs/0.4.parquet",
    "../pqs/0.6.parquet",
    "../pqs/0.8.parquet",
  ];

  let runtime = tokio::runtime::Builder::new_current_thread()
    .build()
    .unwrap();

  for input_file in files {
    let (codes, map) = runtime.block_on(build_column_sketch(input_file)).unwrap();
    let group_name = format!("Benchmark {}", input_file);
    let mut group = criterion.benchmark_group(group_name);
    // Baseline
    group.bench_function(BenchmarkId::new("Baseline", 0), |b| {
      b.iter(|| runtime.block_on(execute_plain_parquet(input_file, 50.0)))
    });

    group.bench_function(BenchmarkId::new("Column Sketch", 0), |b| {
      b.iter(|| runtime.block_on(execute_column_sketch(input_file, 50.0, &codes, &map)))
    });
  }
}

criterion_group!(benches, bench_parquet);
criterion_main!(benches);
