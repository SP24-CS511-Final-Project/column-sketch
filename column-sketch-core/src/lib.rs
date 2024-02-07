//! This crate provides the core data structure and algorithm of [column-sketch](https://stratos.seas.harvard.edu/files/stratos/files/sketches.pdf).
#![allow(dead_code)]

use num_traits::Num;

pub const COMPRESSION_MAP_SIZE: usize = 256;

/// A [`NumericColumnSketch`] is the compression map for numeric column data.
/// It is designed to be space efficient and easily serializable to persistent storage.
/// Under the hood, a [`NumericColumnSketch`] is just a bucket array, where the ith value is the max
/// value belonging to code i, so that a value is mapped to i iff arr[i-1] < value <= arr[i], and a
/// bit vector indicating if the code is unique.
pub struct NumericColumnSketch<T> {
  buckets: Vec<T>,
  unique: Vec<bool>,
}

impl<T: PartialOrd + PartialEq + Copy + Num> NumericColumnSketch<T> {
  /// The implementation of constructing compression map from numeric column values.
  /// Step 1: Handle frequent values.
  ///   - Frequent values are defined as values whose frequency >= 1/C, where C is the size of the compression map (256 for now)
  ///   - Assume input data has size N, then a frequent value would occur at least N/C times, and hence one of the values in sorted
  ///     list (n/c,2n/c,...,(c−1)n/c) must be that frequent value.
  ///     Once a frequent value is found, we find the midpoint position of its occurrence in the sorted list and assign code c * midpoint/n
  ///     to it.
  ///   Caveat:
  ///   1. we don't allow frequent values mapped to consecutive unique code. If both code i and i + 1 has a frequent value, we only give
  ///      unique code to the one with higher frequency.
  ///   2. we don't give unique code MIN or MAX in the code range.
  ///
  /// Step 2: Construct equi-depth histograms between each unique codes
  ///
  pub fn construct(input: Vec<T>) -> NumericColumnSketch<T> {
    let _n = input.len();
    let mut input = input;
    input.sort_by(|t1, t2| t1.partial_cmp(t2).unwrap());

    let _frequent_values = Self::generate_frequent_values(&input, COMPRESSION_MAP_SIZE);

    unimplemented!()
  }

  /// Scan over the input data and generate all eligible frequent values
  ///
  /// * `input`: Sorted input values
  /// * `c`: Number of bucket/code in the compression map
  fn generate_frequent_values(input: &[T], c: usize) -> Vec<FrequentValueEntry> {
    let mut entries = Vec::new();
    let mut start = 0;
    let mut start_value = input[start];

    let threshold = input.len() / c;

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
    if input.len() == 0 {
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

    assert!(buckets.len() <= num_buckets, "Actual constructed bucket length {} > num_buckets {}. Please check the integrity of input data", buckets.len(), num_buckets);

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
      println!("{}, {}", old_code, current_code);
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

#[cfg(test)]
mod tests {
  use std::collections::BTreeSet;

  use rand::{rngs::SmallRng, Rng, SeedableRng};
  use rand_distr::{Distribution, StandardNormal};
  use rstest::rstest;

  use crate::NumericColumnSketch;

  /// White-box testing for frequent value generation.
  #[rstest]
  #[case(1000, 256)]
  #[case(10000, 256)]
  #[case(20000, 256)]
  #[case(20000, 512)]
  fn test_frequent_values_sanity(#[case] dataset_size: usize, #[case] map_size: usize) {
    // frequent_occurrence = Math.floor(n / c)
    let frequent_occurrence = dataset_size / map_size + 1;

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
  fn test_histogram_normal() {
    let mut rng = rand::thread_rng();
    let mut sample: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(100000).collect();
    sample.sort_by(|f1, f2| f1.partial_cmp(f2).unwrap());

    let histogram = NumericColumnSketch::generate_equi_depth_histogram(&sample, 1000, 100, f64::MAX);
    println!("{:?}", histogram);
  }
}
