//! The parquet benchmark experiment is based on the physical execution design of [arrow-datafusion](https://github.com/apache/arrow-datafusion) to simulate a realistic workflow
//! Arrow-datafusion passes `Vec<RecordBatch>` between operators, which means filter operator materializes the result rather than passing the selection list or bitvector
//! along the execution. The experiment implements the same interface.
//! The core experiment interface is a function:
//! `fn execute(path: impl Path, target_value: f64) -> Pin<Box<dyn Stream<Item=Result<RecordBatch>> + Send>>`, which accepts a parquet file and a target predicate value
//! and returns an async stream of `RecordBatch`s, which roughly matches the [`execute` API from datafusion](https://github.com/apache/arrow-datafusion/blob/1e4ddb6d86328cb6596bb50da9ccc654f19a83ea/datafusion/physical-plan/src/lib.rs#L378C5-L382)
//! The benchmark harness will create and drain the stream as a complete experiment run.

// Some of the code is adapted from [arrow-datafusion](https://github.com/apache/arrow-datafusion)

use std::{path::Path, sync::Arc};

use anyhow::Result;
use arrow::{
  array::{Array, AsArray, PrimitiveArray, RecordBatch, RecordBatchOptions},
  datatypes::Float64Type,
};
use futures::StreamExt;
use parquet::{arrow::ParquetRecordBatchStreamBuilder, file::statistics::Statistics::Double};

/// Return all the records greater than target
pub async fn execute_plain_parquet(
  path: impl AsRef<Path>,
  target: f64,
) -> Result<Vec<RecordBatch>> {
  let file = tokio::fs::File::open(path).await?;

  let stream_builder = ParquetRecordBatchStreamBuilder::new(file).await?;

  let metadata = stream_builder.metadata();

  // Step 1: Perform Zone-map pruning on rowgroups
  let pruned_rowgroups = {
    let mut rowgroups = Vec::with_capacity(metadata.num_row_groups());
    for (idx, rowgroup_metadata) in metadata.row_groups().iter().enumerate() {
      let column_metadata = &rowgroup_metadata.columns()[0];
      if let Some(statistics) = column_metadata.statistics() {
        match statistics {
          Double(s) => {
            if *s.min() >= target {
              continue;
            }
          }
          _ => unreachable!(),
        }
      }
      rowgroups.push(idx);
    }
    rowgroups
  };

  let mut stream = stream_builder.with_row_groups(pruned_rowgroups).build()?;

  let mut result = vec![];

  // Step 2: Perform value level pruning
  while let Some(next_batch) = stream.next().await {
    let next_batch = next_batch?;
    let column = next_batch.column(0);
    let f64_array = column.as_primitive::<Float64Type>();
    let filter_result = f64_array
      .iter()
      .map(|val| val.unwrap())
      .filter(|value| *value > target);
    let filtered_array: PrimitiveArray<Float64Type> =
      PrimitiveArray::from_iter_values(filter_result);
    let row_count = filtered_array.len();
    let filtered_columns: Vec<Arc<dyn Array>> = vec![Arc::new(filtered_array)];
    let options = RecordBatchOptions::default().with_row_count(Some(row_count));
    let new_record_batch =
      RecordBatch::try_new_with_options(next_batch.schema(), filtered_columns, &options)?;
    result.push(new_record_batch);
  }

  Ok(result)
}

#[cfg(test)]
mod tests {
  use arrow::util::pretty::print_batches;

  use super::*;

  #[tokio::test]
  async fn test_scan_parquet() {
    let batches = execute_plain_parquet("../example.parquet", 50.0)
      .await
      .unwrap();
    print_batches(&batches).unwrap();
  }
}
