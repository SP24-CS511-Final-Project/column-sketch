use column_sketch_core::NumericColumnSketch;
use rstest::rstest;

use crate::{
  column_sketch_plain_pred_greater, column_sketch_simd_pred_greater, naive_plain_pred_greater,
  util::gen_dataset_normal,
};

#[rstest]
#[case(1000)]
#[case(10000)]
#[case(50000)]
#[case(100000)]
fn test_eval_correctness(#[case] size: usize) {
  let sample = gen_dataset_normal(size);
  let sorted_sample = {
    let mut sorted = sample.clone();
    sorted.sort_by(|f1, f2| f1.partial_cmp(f2).unwrap());
    sorted
  };

  let column_sketch = NumericColumnSketch::construct_from_sorted(&sorted_sample);

  let compressed_code = column_sketch.compress_array(&sample);

  let target_idx = size / 2;
  let target_value = sorted_sample[target_idx];

  // Baseline
  let expected = eval_plain_get_result(&sample, target_value, naive_plain_pred_greater);

  // Column Sketch Plain
  let column_plain = eval_column_sketch_get_result(
    &sample,
    &column_sketch,
    &compressed_code,
    target_value,
    column_sketch_plain_pred_greater,
  );

  assert_eq!(expected, column_plain);

  // Column SIMD
  let column_simd = eval_column_sketch_get_result(
    &sample,
    &column_sketch,
    &compressed_code,
    target_value,
    column_sketch_simd_pred_greater,
  );

  assert_eq!(expected, column_simd);
}

fn eval_column_sketch_get_result<F>(
  sample: &[f64],
  compress_map: &NumericColumnSketch<f64>,
  compressed_code: &[u8],
  target: f64,
  f: F,
) -> Vec<bool>
where
  F: Fn(&[f64], &NumericColumnSketch<f64>, &[u8], f64, &mut [bool]),
{
  let n = sample.len();
  let mut output = vec![false; n];
  f(sample, compress_map, compressed_code, target, &mut output);
  output
}

fn eval_plain_get_result<F>(sample: &[f64], target: f64, f: F) -> Vec<bool>
where
  F: Fn(&[f64], f64, &mut [bool]),
{
  let n = sample.len();
  let mut output = vec![false; n];
  f(sample, target, &mut output);
  output
}
