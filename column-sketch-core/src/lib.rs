//! This crate provides the core data structure and algorithm of [column-sketch](https://stratos.seas.harvard.edu/files/stratos/files/sketches.pdf).
#![allow(dead_code)]
#![feature(portable_simd)]

pub mod traits;

use std::cmp::Ordering;

use traits::Numeric;

pub const COMPRESSION_MAP_SIZE: usize = 256;

/// A [`NumericColumnSketch`] is the compression map for numeric column data.
/// It is designed to be space efficient and easily serializable to persistent storage.
/// Under the hood, a [`NumericColumnSketch`] is just a bucket array, where the ith value is the max
/// value belonging to code i, so that a value is mapped to i iff arr[i-1] < value <= arr[i], and a
/// bit vector indicating if the code is unique.
#[derive(Debug)]
pub struct NumericColumnSketch<T> {
  pub buckets: Vec<T>,
  pub unique: Vec<bool>,
}

impl<T: Numeric> NumericColumnSketch<T> {
  /// Compress the given value using the column sketch compression map.
  ///
  /// The algorithm for compression using column sketch runs as follows:
  /// 1. Find the smallest index i where buckets[i] > value
  /// 2. If unique[i]: return i - 1
  /// 3. If buckets[i - 1] == value: return i - 1
  /// 3. Else: return i
  /// and our value lies between the frequent value and its prev bucket's max value.
  /// In this case the value goes to the previous bucket)
  pub fn compress(&self, value: T) -> usize {
    // index points to the first element which is greater than value
    let index = self
      .buckets
      .binary_search_by(|element| match element.partial_cmp(&value).unwrap() {
        Ordering::Equal => Ordering::Less,
        ord => ord,
      })
      .unwrap_err();

    if index == 0 {
      return index;
    }

    // Example: buckets = [10, 20, 30, 40, 50], unqiue = [false, true, false, false, false], value = 19
    // index will be 1, which points to 20, but since 1 is the unique code for value 20, 19 actually goes to code 0
    let prev_index = index - 1;
    if self.unique[index] {
      return prev_index;
    }

    let prev_value = self.buckets[prev_index];
    if prev_value == value {
      return prev_index;
    }

    // buckets[index - 1] < value < buckets[index], value -> index
    index
  }

  pub fn compress_array(&self, input: &[T]) -> Vec<u8> {
    input
      .iter()
      .map(|value| self.compress(*value) as u8)
      .collect()
  }

  /// Return if a given code is a unique code
  pub fn is_unique_code(&self, code: usize) -> bool {
    self.unique[code]
  }

  /// The implementation of constructing compression map from numeric column values.
  /// Step 1: Handle frequent values.
  ///   - Frequent values are defined as values whose frequency >= 1/C, where C is the size of the compression map (256 as a default so that code fits in an 8-bit integer)
  ///   - Assume input data has size N, then a frequent value would occur at least N/C times.
  ///     Once a frequent value is found, we find the midpoint position of its occurrence in the sorted list and assign code c * midpoint/n
  ///     to it.
  ///   Caveat:
  ///   1. we don't allow frequent values mapped to consecutive unique code. If both code i and i + 1 has a frequent value, we only give
  ///      unique code to the one with higher frequency.
  ///   2. we don't give unique code MIN or MAX in the code range (0 and 255 in the default setting).
  ///
  /// Step 2: Construct equi-depth histograms between each unique codes
  ///
  pub fn construct(input: Vec<T>) -> NumericColumnSketch<T> {
    let mut input = input;
    input.sort_by(|t1, t2| t1.partial_cmp(t2).unwrap());
    Self::construct_from_sorted(&input)
  }

  pub fn construct_from_sorted(sorted_input: &[T]) -> NumericColumnSketch<T> {
    let input = sorted_input;
    let n = input.len();
    let value_per_bucket = frequent_threshold(n, COMPRESSION_MAP_SIZE);

    // Step 1: Decide frequent values -> unique codes
    let frequent_values = Self::generate_frequent_values(input, COMPRESSION_MAP_SIZE);

    let mut buckets = vec![T::max_value(); COMPRESSION_MAP_SIZE];
    let mut unique = vec![false; COMPRESSION_MAP_SIZE];

    // Step 2: Construct equi-depth histogram between each frequent values
    let mut cur_start = 0;
    let mut last_code = 0;
    for entry in frequent_values {
      let value = input[entry.start];
      let code = entry.code(n, COMPRESSION_MAP_SIZE);

      // Construct histogram between the current entry and the last entry
      let histogram_before_entry = Self::generate_equi_depth_histogram(
        &input[cur_start..entry.start],
        code - last_code,
        value_per_bucket,
        value,
      );
      (buckets[last_code..code]).copy_from_slice(&histogram_before_entry);

      // Set current code
      buckets[code] = value;
      unique[code] = true;

      last_code = code + 1;
      cur_start = entry.end;
    }

    // Construct last run of histogram
    let histogram_before_entry = Self::generate_equi_depth_histogram(
      &input[cur_start..],
      COMPRESSION_MAP_SIZE - last_code,
      value_per_bucket,
      T::max_value(),
    );
    (buckets[last_code..]).copy_from_slice(&histogram_before_entry);

    NumericColumnSketch { buckets, unique }
  }

  /// Scan over the input data and generate all eligible frequent values
  ///
  /// * `input`: Sorted input values
  /// * `c`: Number of bucket/code in the compression map
  fn generate_frequent_values(input: &[T], c: usize) -> Vec<FrequentValueEntry> {
    let mut entries = Vec::new();
    let mut start = 0;
    let mut start_value = input[start];

    let threshold = frequent_threshold(input.len(), c);

    for (end, &value) in input.iter().enumerate() {
      if value != start_value {
        let occurrence = end - start;
        if occurrence >= threshold {
          add_new_entry(
            &mut entries,
            FrequentValueEntry { start, end },
            input.len(),
            c,
          );
        }
        start = end;
        start_value = value;
      }
    }

    // Handle the last value
    let occurrence = input.len() - start;
    if occurrence >= threshold {
      add_new_entry(
        &mut entries,
        FrequentValueEntry {
          start,
          end: input.len(),
        },
        input.len(),
        c,
      );
    }

    entries
  }

  /// This function generates a histogram of given number of bucket from a slice of data, and is used for
  /// generating histograms over slices of data between each frequent values.
  ///
  /// Invariant:
  /// The input data is always between two frequent values i and j.
  /// num_buckets = Code[j] - Code[i] - 1 >= 1
  /// value_per_bucket = n / c
  /// end_value = code[j], or T::MAX if the last bucket
  ///
  /// Returns:
  /// A vector histogram of size num_buckets
  fn generate_equi_depth_histogram(
    input: &[T],
    num_buckets: usize,
    value_per_bucket: usize,
    end_value: T,
  ) -> Vec<T> {
    // Consider an extreme case:
    // n = 25600, c = 256 (hence value_per_bucket = 100)
    // index [0-200) are value 1 -> code 1
    // index [201-3000] are value 10 -> code 16
    // between these two frequent values, there is [2-15], which is 14 buckets
    // but no values.
    // In this case we fill every bucket with 10,
    // and when assigning values, we will do a special casing on frequent values to distinguish
    // the buckets.
    if input.is_empty() {
      return vec![end_value; num_buckets];
    }

    let mut buckets = Vec::with_capacity(num_buckets);

    let mut prev_value = input[0];
    let mut prev_occurrence = 1;
    for &value in input.iter().skip(1) {
      if value == prev_value {
        prev_occurrence += 1;
        continue;
      }

      if prev_occurrence >= value_per_bucket {
        buckets.push(prev_value);
        prev_value = value;
        prev_occurrence = 1;
      } else {
        prev_value = value;
        prev_occurrence += 1;
      }
    }

    // Handle last values
    if prev_occurrence >= value_per_bucket {
      buckets.push(prev_value);
    }

    assert!(buckets.len() <= num_buckets, "Actual constructed bucket length {} > num_buckets {} (value_per_bucket={}). Please check the integrity of input data", buckets.len(), num_buckets, value_per_bucket);

    if buckets.len() == num_buckets {
      *buckets.last_mut().unwrap() = end_value;
    } else {
      while buckets.len() < num_buckets {
        buckets.push(end_value);
      }
    }

    buckets
  }
}

#[derive(Debug, Clone, Copy)]
struct FrequentValueEntry {
  /// input[start] is the first occurrence of the value
  pub start: usize,
  /// input[end] is one past the last occurrence of the value
  pub end: usize,
}

impl FrequentValueEntry {
  pub fn code(self, n: usize, c: usize) -> usize {
    let midpoint = (self.start + self.end) / 2;
    ((c as f64 * midpoint as f64) / n as f64).round() as usize
  }
}

/// Add a new frequent value while enforcing constraints
fn add_new_entry(
  entries: &mut Vec<FrequentValueEntry>,
  new_entry: FrequentValueEntry,
  n: usize,
  c: usize,
) {
  let current_code = new_entry.code(n, c);
  if current_code == 0 || current_code == c - 1 {
    return;
  }

  match entries.last().copied() {
    Some(old_entry) => {
      let old_code = old_entry.code(n, c);
      if old_code == current_code - 1 || old_code == current_code {
        let old_freq = old_entry.end - old_entry.start;
        let new_freq = new_entry.end - new_entry.start;
        if new_freq > old_freq {
          entries.pop();
          entries.push(new_entry);
        }
      } else {
        entries.push(new_entry);
      }
    }
    None => entries.push(new_entry),
  }
}

// The threshold for a frequent value: Math.ceil(n / c)
fn frequent_threshold(n: usize, c: usize) -> usize {
  (n as f64 / c as f64).ceil() as usize
}

#[cfg(test)]
mod tests {
  use std::collections::{BTreeMap, BTreeSet};

  use rand::{rngs::SmallRng, Rng, SeedableRng};
  use rand_distr::{Distribution, StandardNormal};
  use rstest::rstest;

  use crate::{frequent_threshold, traits::Numeric, NumericColumnSketch, COMPRESSION_MAP_SIZE};

  /// White-box testing for frequent value generation.
  #[rstest]
  #[case(1000, 256)]
  #[case(10000, 256)]
  #[case(20000, 256)]
  #[case(20000, 512)]
  fn test_frequent_values_sanity(#[case] dataset_size: usize, #[case] map_size: usize) {
    let frequent_occurrence = frequent_threshold(dataset_size, map_size);

    let mut rng = SmallRng::seed_from_u64(64);

    let mut data = Vec::new();
    let mut current_value = 0;
    let mut is_frequent = BTreeSet::new();
    let mut last_frequent_code = 0;

    while data.len() < dataset_size {
      let current_start = data.len();

      let current_code_if_frequent = (map_size as f64
        * ((current_start + current_start + frequent_occurrence) / 2) as f64
        / dataset_size as f64)
        .round() as usize;

      let should_generate_frequent = (0..map_size - 1).contains(&current_code_if_frequent)
        && current_code_if_frequent > last_frequent_code + 1
        && rng.gen_ratio(1, 3);

      if should_generate_frequent {
        is_frequent.insert(current_value);
        data.extend(std::iter::repeat(current_value).take(frequent_occurrence));
        last_frequent_code = current_code_if_frequent;
      } else {
        data.push(current_value);
      }

      current_value += 1;
    }

    let entries = NumericColumnSketch::generate_frequent_values(&data, map_size);
    let generated_frequent_value: BTreeSet<i32> =
      entries.iter().map(|entry| data[entry.start]).collect();

    assert_eq!(is_frequent, generated_frequent_value);
  }

  #[test]
  fn test_data_less_than_bucket() {
    // In case where c = 256 and n = 100, every value except the first 1 gets a unique code
    // Example:
    // 1 has start 1 and end 2, midpoint is 1, and hence code = 256 / 100 = 2
    // 2 has start 2 and end 3, midpoint is 2, and code = 256 * 2 / 100 = 5
    // ...

    let c = 256;
    let data: Vec<i32> = (0..=100).into_iter().collect();
    let entries = NumericColumnSketch::generate_frequent_values(&data, c);
    let frequent_values: Vec<i32> = entries.iter().map(|entry| data[entry.start]).collect();

    let expected_values: Vec<i32> = (1..=100).into_iter().collect();
    assert_eq!(expected_values, frequent_values);
  }

  #[test]
  fn test_histogram_simple() {
    let input = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
    let histogram = NumericColumnSketch::generate_equi_depth_histogram(&input, 4, 4, 100);
    assert_eq!(vec![1, 2, 3, 100], histogram);
  }

  #[test]
  fn test_histogram_no_input() {
    let input = vec![];
    let histogram = NumericColumnSketch::generate_equi_depth_histogram(&input, 10, 5, 100);
    assert_eq!(vec![100; 10], histogram);
  }

  #[test]
  fn test_histogram_unique() {
    let input: Vec<i32> = (0..=1000).into_iter().collect();
    let histogram = NumericColumnSketch::generate_equi_depth_histogram(&input, 100, 20, 2000);

    // We should fill histogram with 19, 39, ... 999 followed by all 2000
    let mut expected: Vec<i32> = (19..=999).step_by(20).into_iter().collect();
    while expected.len() < 100 {
      expected.push(2000);
    }
    assert_eq!(expected, histogram);
  }

  #[test]
  fn test_column_sketch_order_preserving() {
    let mut rng = rand::thread_rng();
    // Sample is normally distributed f64 data with no frequent values.
    let sample: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(1000).collect();

    let column_sketch = NumericColumnSketch::construct(sample.clone());

    verify_order_preserving(&column_sketch, &sample);
  }

  #[test]
  fn test_column_sketch_with_frequent_values() {
    // The dataset is constructed as follows:
    // every data point represents a student's score ranging from 0 - 1000
    // student has and independent probability 40% of getting 600, 20% of getting 700, 15% of getting 800,
    // 5% of getting 900, and then the remaining 20% getting any rate uniformly in range [0 - 1000]
    // we sample 10000 values and expect at least 600, 700, 800, and 900 to be frequent values.
    let mut rng = rand::thread_rng();
    let mut sample = Vec::with_capacity(10000);
    for _ in 0..10000 {
      let dice = rng.gen_range(0..100);
      if dice < 40 {
        sample.push(600);
      } else if dice < 60 {
        sample.push(700);
      } else if dice < 75 {
        sample.push(800);
      } else if dice < 80 {
        sample.push(900);
      } else {
        sample.push(rng.gen_range(0..=1000));
      }
    }

    let mut statistics: BTreeMap<i32, i32> = BTreeMap::new();
    for &v in &sample {
      *statistics.entry(v).or_default() += 1;
    }

    let column_sketch = NumericColumnSketch::construct(sample.clone());

    verify_order_preserving(&column_sketch, &sample);

    // verify unique value
    let threshold = frequent_threshold(10000, COMPRESSION_MAP_SIZE);
    for (value, occurrence) in &statistics {
      if *occurrence >= threshold as i32 {
        let code: usize = column_sketch.compress(*value);
        // println!("Unique code {} for frequent value {}", code, value);
        assert!(column_sketch.is_unique_code(code));
      } else {
        let code: usize = column_sketch.compress(*value);
        assert!(!column_sketch.is_unique_code(code))
      }
    }
  }

  fn verify_order_preserving<T: Numeric>(column_sketch: &NumericColumnSketch<T>, sample: &[T]) {
    let codes: Vec<u8> = sample
      .iter()
      .map(|value| column_sketch.compress(*value) as u8)
      .collect();
    // Check that the map is order-preserving.
    for (i, n1) in sample.iter().enumerate() {
      for (j, n2) in sample.iter().enumerate() {
        if n1 > n2 {
          assert!(codes[i] >= codes[j]);
        } else if n1 == n2 {
          assert!(codes[i] == codes[j])
        } else {
          assert!(codes[i] <= codes[j])
        }
      }
    }
  }
}
