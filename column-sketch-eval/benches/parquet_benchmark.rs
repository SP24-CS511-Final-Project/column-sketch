//! The parquet benchmark experiment is based on the physical execution design of [arrow-datafusion](https://github.com/apache/arrow-datafusion) to simulate a realistic workflow
//! Arrow-datafusion passes `Vec<RecordBatch>` between operators, which means filter operator materializes the result rather than passing the selection list or bitvector
//! along the execution. The experiment implements the same interface.
//! The core experiment interface is a function:
//! `fn execute(path: impl Path, target_value: f64) -> Pin<Box<dyn Stream<Item=Result<RecordBatch>> + Send>>`, which accepts a parquet file and a target predicate value
//! and returns an async stream of `RecordBatch`s, which roughly matches the [`execute` API from datafusion](https://github.com/apache/arrow-datafusion/blob/1e4ddb6d86328cb6596bb50da9ccc654f19a83ea/datafusion/physical-plan/src/lib.rs#L378C5-L382)
//! The benchmark harness will create and drain the stream as a complete experiment run.
