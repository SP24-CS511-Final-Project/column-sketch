//! The parquet benchmark experiment is based on the physical execution design of [arrow-datafusion](https://github.com/apache/arrow-datafusion) to simulate a realistic workflow
//! Arrow-datafusion passes `Vec<RecordBatch>` between operators, which means filter operator materializes the result rather than passing the selection list or bitvector
//! along the execution. The experiment implements the same interface.
//! The core experiment interface is a function:
//! `fn execute(path: impl Path, target_value: f64) -> Pin<Box<dyn Stream<Item=Result<RecordBatch>> + Send>>`, which accepts a parquet file and a target predicate value
//! and returns an async stream of `RecordBatch`s, which roughly matches the [`execute` API from datafusion](https://github.com/apache/arrow-datafusion/blob/1e4ddb6d86328cb6596bb50da9ccc654f19a83ea/datafusion/physical-plan/src/lib.rs#L378C5-L382)
//! The benchmark harness will create and drain the stream as a complete experiment run.

// Some of the code is adapted from [arrow-datafusion](https://github.com/apache/arrow-datafusion)

use std::{
  path::Path,
  simd::{
    cmp::{SimdPartialEq, SimdPartialOrd},
    Simd,
  },
  sync::Arc,
};

use anyhow::Result;
use arrow::{
  array::{
    Array, AsArray, BooleanArray, BooleanBufferBuilder, Float64Array, MutableArrayData,
    PrimitiveArray, RecordBatch, RecordBatchOptions,
  },
  datatypes::Float64Type,
};
use column_sketch_core::NumericColumnSketch;
use futures::StreamExt;
use parquet::{arrow::ParquetRecordBatchStreamBuilder, file::statistics::Statistics::Double};

use crate::{SIMD_REGISTER_WIDTH, U8_WIDTH};

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
            if *s.max() <= target {
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

pub async fn build_column_sketch(
  path: impl AsRef<Path>,
) -> Result<(Vec<Vec<u8>>, NumericColumnSketch<f64>)> {
  let file = tokio::fs::File::open(path).await?;
  let stream_builder = ParquetRecordBatchStreamBuilder::new(file).await?;
  let mut stream = stream_builder.build()?;

  let mut batches = vec![];
  let mut result: Vec<f64> = vec![];

  // Step 2: Perform value level pruning
  while let Some(next_batch) = stream.next().await {
    let next_batch = next_batch?;
    let column = next_batch.column(0);
    let f64_array = column.as_primitive::<Float64Type>();
    result.extend(f64_array.values());
    batches.push(next_batch);
  }

  let column_sketch = NumericColumnSketch::construct(result.clone());

  let compressed_code = batches
    .into_iter()
    .map(|batch| {
      let array: Vec<f64> = batch
        .column(0)
        .as_primitive::<Float64Type>()
        .values()
        .iter()
        .copied()
        .collect();
      let code = column_sketch.compress_array(&array);
      code
    })
    .collect();

  Ok((compressed_code, column_sketch))
}

pub async fn execute_column_sketch(
  path: impl AsRef<Path>,
  target: f64,
  codes: &Vec<Vec<u8>>,
  compression_map: &NumericColumnSketch<f64>,
) -> Result<Vec<RecordBatch>> {
  let target_code = compression_map.compress(target) as u8;
  let file = tokio::fs::File::open(path).await?;

  let stream_builder = ParquetRecordBatchStreamBuilder::new(file).await?;
  let mut stream = stream_builder.build()?;

  let mut result = vec![];

  // Step 2: Perform value level pruning
  let mut idx = 0;
  while let Some(next_batch) = stream.next().await {
    let next_batch = next_batch?;

    let bitvector = build_boolean_array(&next_batch, &codes[idx], target_code, target);

    // Build a new record batch based on the boolean array
    let data = next_batch.column(0).to_data();
    let mut mutable = MutableArrayData::new(vec![&data], false, codes[idx].len());
    bitvector
      .values()
      .set_slices()
      .for_each(|(start, end)| mutable.extend(0, start, end));

    let new_data = mutable.freeze();
    let new_array = Arc::new(Float64Array::from(new_data));
    let options = RecordBatchOptions::default().with_row_count(Some(new_array.len()));
    let new_batch =
      RecordBatch::try_new_with_options(next_batch.schema(), vec![new_array], &options)?;

    result.push(new_batch);
    idx += 1;
  }

  Ok(result)
}

pub fn build_boolean_array(
  batch: &RecordBatch,
  code: &[u8],
  target_code: u8,
  target: f64,
) -> BooleanArray {
  let n = code.len();
  let mut builder = BooleanBufferBuilder::new(n);
  builder.resize(n);
  column_sketch_simd_pred_greater(
    batch.column(0).as_primitive(),
    code,
    target,
    target_code,
    &mut builder,
  );
  BooleanArray::new(builder.finish(), None)
}

fn column_sketch_plain_pred_greater_arrow(
  original_input: &PrimitiveArray<Float64Type>,
  compressed_code: &[u8],
  target: f64,
  target_code: u8,
  output: &mut BooleanBufferBuilder,
  base: usize,
) {
  compressed_code.iter().enumerate().for_each(|(idx, &code)| {
    if code != target_code {
      output.set_bit(base + idx, code > target_code);
    } else {
      output.set_bit(base + idx, original_input.value(idx) > target);
    }
  });
}

fn column_sketch_simd_pred_greater(
  original_input: &PrimitiveArray<Float64Type>,
  compressed_code: &[u8],
  target: f64,
  target_code: u8,
  output: &mut BooleanBufferBuilder,
) {
  const SEGMENT_SIZE: usize = SIMD_REGISTER_WIDTH / U8_WIDTH;

  let n = original_input.len();
  let num_segment = n / SEGMENT_SIZE;
  let risidual_start = num_segment * SEGMENT_SIZE;

  if risidual_start < n {
    column_sketch_plain_pred_greater_arrow(
      &original_input.slice(risidual_start, n - risidual_start),
      &compressed_code[risidual_start..],
      target,
      target_code,
      output,
      risidual_start,
    );
  }

  let target_simd = Simd::from([target_code; SEGMENT_SIZE]);

  let mut output_position = 0;

  // temp_result holds positions that we are not sure of
  let mut temp_result = Vec::with_capacity(128);

  for i in 0..num_segment {
    let idx = i * SEGMENT_SIZE;
    let code_simd: Simd<u8, SEGMENT_SIZE> =
      Simd::from_slice(&compressed_code[idx..idx + SEGMENT_SIZE]);
    let definitely_greater = code_simd.simd_gt(target_simd);
    let possibly_greater = code_simd.simd_eq(target_simd);

    // Store definite answers
    for (idx, value) in definitely_greater.to_array().into_iter().enumerate() {
      output.set_bit(output_position + idx, value);
    }

    for i in 0..SEGMENT_SIZE {
      if possibly_greater.test(i) {
        temp_result.push(idx + i);
      }
    }

    output_position += SEGMENT_SIZE;
  }

  for position in temp_result {
    output.set_bit(position, original_input.value(position) > target);
  }
}

#[cfg(test)]
mod tests {
  use plotters::{
    backend::BitMapBackend,
    chart::ChartBuilder,
    coord::{combinators::IntoLinspace, ranged1d::DiscreteRanged},
    drawing::IntoDrawingArea,
    element::PathElement,
    series::LineSeries,
    style::{Color, IntoFont, BLACK, BLUE, RED, WHITE},
  };

  use super::*;

  #[tokio::test]
  async fn test_scan_parquet() {
    let batches = execute_plain_parquet("../pqs/0.8.parquet", 50.0)
      .await
      .unwrap();
    // print_batches(&batches).unwrap();

    let (codes, compression_map) = build_column_sketch("../pqs/0.8.parquet").await.unwrap();
    let abatches = execute_column_sketch("../pqs/0.8.parquet", 50.0, &codes, &compression_map)
      .await
      .unwrap();
    // print_batches(&abatches).unwrap();
    check_batches(&batches, &abatches);
  }

  fn check_batches(expected: &[RecordBatch], actual: &[RecordBatch]) {
    let expected_values: Vec<f64> = expected
      .into_iter()
      .flat_map(|batch| batch.column(0).as_primitive::<Float64Type>().values())
      .copied()
      .collect();
    let actual_values: Vec<f64> = actual
      .into_iter()
      .flat_map(|batch| batch.column(0).as_primitive::<Float64Type>().values())
      .copied()
      .collect();
    assert_eq!(expected_values, actual_values);
  }

  #[tokio::test]
  async fn test_rowgroup_filter_effectiveness() {
    async fn get_rowgroup_filter_percentage(path: impl AsRef<Path>, predicate: f64) -> f64 {
      let file = tokio::fs::File::open(path).await.unwrap();

      let stream_builder = ParquetRecordBatchStreamBuilder::new(file).await.unwrap();

      let metadata = stream_builder.metadata();

      let original_rowgroups = metadata.row_groups().len();

      // Step 1: Perform Zone-map pruning on rowgroups
      let pruned_rowgroups = {
        let mut rowgroups = Vec::with_capacity(metadata.num_row_groups());
        for (idx, rowgroup_metadata) in metadata.row_groups().iter().enumerate() {
          let column_metadata = &rowgroup_metadata.columns()[0];
          if let Some(statistics) = column_metadata.statistics() {
            match statistics {
              Double(s) => {
                if *s.min() >= predicate {
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

      pruned_rowgroups.len() as f64 / original_rowgroups as f64
    }

    let sortedness = [0.2f64, 0.4, 0.6, 0.8];
    let paths = [
      "../pqs/0.2.parquet",
      "../pqs/0.4.parquet",
      "../pqs/0.6.parquet",
      "../pqs/0.8.parquet",
    ];
    let mut percentage = Vec::new();
    for p in paths {
      percentage.push(get_rowgroup_filter_percentage(p, 500.0).await);
    }

    // Plot
    let root = BitMapBackend::new("../assets/zone-map.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
      .caption(
        "Zone map pruning effectiveness",
        ("sans-serif", 30).into_font(),
      )
      .margin(10)
      .x_label_area_size(50)
      .y_label_area_size(50)
      .build_cartesian_2d(0f32..1f32, 0f32..1f32)
      .unwrap();

    chart
      .configure_mesh()
      .x_desc("Sorted-ness")
      .axis_desc_style(("sans-serif", 20, &BLACK))
      .y_desc("Prunning rate (Lower the better)")
      .draw()
      .unwrap();

    chart
      .draw_series(LineSeries::new(
        sortedness
          .into_iter()
          .take(4)
          .zip(percentage.iter())
          .map(|(x, y)| (x as f32, *y as f32)),
        &RED,
      ))
      .unwrap();

    chart
      .configure_series_labels()
      .background_style(&WHITE.mix(0.8))
      .border_style(&BLACK)
      .draw()
      .unwrap();

    root.present().unwrap();
  }

  #[tokio::test]
  async fn plot() {
    let root_area = BitMapBackend::new("../assets/endtoend.png", (1024, 768)).into_drawing_area();

    root_area.fill(&WHITE).unwrap();

    let root_area = root_area
      .titled("Filter + Scan Performance", ("sans-serif", 30))
      .unwrap();

    let sortedness = [0.2f32, 0.4, 0.6, 0.8];
    let baselines = [4.3972f32, 4.3452, 4.29, 4.15];
    let cs = [4.3141f32, 4.3152, 4.4871, 4.4203];

    let mut cc = ChartBuilder::on(&root_area)
      .margin(5)
      .set_all_label_area_size(50)
      .caption("Zone Map and Column Sketch", ("sans-serif", 40))
      .build_cartesian_2d(0.0f32..1.0, 4.0f32..4.532)
      .unwrap();

    cc.configure_mesh()
      .x_labels(20)
      .y_labels(10)
      .x_desc("Sorted-ness")
      .axis_desc_style(("sans-serif", 20, &BLACK))
      .y_desc("Processing time (the lower the better)")
      .disable_mesh()
      .x_label_formatter(&|v| format!("{:.1}", v))
      .y_label_formatter(&|v| format!("{:.1}", v))
      .draw()
      .unwrap();

    cc.draw_series(LineSeries::new(
      sortedness
        .iter()
        .take(4)
        .zip(baselines.iter().take(4))
        .map(|(x, y)| (*x, *y)),
      &RED,
    ))
    .unwrap()
    .label("Baseline (Parquet Zone-map)")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    cc.draw_series(LineSeries::new(
      sortedness
        .iter()
        .take(4)
        .zip(cs.iter().take(4))
        .map(|(x, y)| (*x, *y)),
      &BLUE,
    ))
    .unwrap()
    .label("Column Sketch")
    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    cc.configure_series_labels()
      .border_style(BLACK)
      .draw()
      .unwrap();

    // To avoid the IO failure being ignored silently, we
    // manually call the present function
    root_area.present().expect("Unable to write result to file, please make sure 'plotters-doc-data' dir exists under current dir");
  }
}
