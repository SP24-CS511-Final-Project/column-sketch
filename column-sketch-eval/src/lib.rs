//! This crate implements evaluation infrastructures for the project, including:
//! - SIMD/Non-SIMD scanner implementation based on column sketch
//! - Micro Benchmark Harness
//! - Other utilities
#![feature(portable_simd)]
#![feature(generic_arg_infer)]

#[cfg(test)]
mod test;
pub mod util;

use std::simd::{
  cmp::{SimdPartialEq, SimdPartialOrd},
  Simd,
};

use column_sketch_core::{traits::Numeric, NumericColumnSketch};

// My system only has AVX1.0, so use 256-bit register width
const SIMD_REGISTER_WIDTH: usize = 256;
const U8_WIDTH: usize = std::mem::size_of::<u8>() * 8;

/// Algorithm: Column Sketch without explicit SIMD
/// Predicate: Select if value > target
/// * `original_input`: original input values
/// * `compression_map`: column sketch compression map
/// * `compressed_code`: pre-materialized code of the orignal input values
/// * `target`: predicate target
/// * `output`: Output bit vector to write the result to
pub fn column_sketch_plain_pred_greater<T: Numeric>(
  original_input: &[T],
  compression_map: &NumericColumnSketch<T>,
  compressed_code: &[u8],
  target: T,
  output: &mut [bool],
) {
  let target_code = compression_map.compress(target) as u8;
  compressed_code.iter().enumerate().for_each(|(idx, &code)| {
    if code != target_code {
      output[idx] = code > target_code;
    } else {
      output[idx] = original_input[idx] > target;
    }
  });
}

/// Algorithm: Column Sketch with explicit SIMD
/// Predicate: Select if value > target
/// * `original_input`: original input values
/// * `compression_map`: column sketch compression map
/// * `compressed_code`: pre-materialized code of the orignal input values
/// * `target`: predicate target
/// * `output`: Output bit vector to write the result to
pub fn column_sketch_simd_pred_greater<T: Numeric>(
  original_input: &[T],
  compression_map: &NumericColumnSketch<T>,
  compressed_code: &[u8],
  target: T,
  output: &mut [bool],
) {
  const SEGMENT_SIZE: usize = SIMD_REGISTER_WIDTH / U8_WIDTH;

  let n = original_input.len();
  let num_segment = n / SEGMENT_SIZE;
  let risidual_start = num_segment * SEGMENT_SIZE;

  if risidual_start < n {
    column_sketch_plain_pred_greater(
      &original_input[risidual_start..],
      compression_map,
      &compressed_code[risidual_start..],
      target,
      &mut output[risidual_start..],
    );
  }

  let target_code = compression_map.compress(target) as u8;
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
    let out_ref = &mut output[output_position..output_position + SEGMENT_SIZE];
    out_ref.copy_from_slice(&definitely_greater.to_array());

    for i in 0..SEGMENT_SIZE {
      if possibly_greater.test(i) {
        temp_result.push(idx + i);
      }
    }

    output_position += SEGMENT_SIZE;
  }

  for position in temp_result {
    output[position] = original_input[position] > target;
  }
}

/// Algorithm: Naive comparison
/// Predicate: Select if value > target
/// * `original_input`: original input values
/// * `target`: predicate target
/// * `output`: Output bit vector to write the result to
pub fn naive_plain_pred_greater<T: Numeric>(original_input: &[T], target: T, output: &mut [bool]) {
  original_input.iter().enumerate().for_each(|(idx, value)| {
    if *value > target {
      output[idx] = true;
    }
  });
}
