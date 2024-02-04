//! This crate provides the core data structure and algorithm of [column-sketch](https://stratos.seas.harvard.edu/files/stratos/files/sketches.pdf).
#![allow(dead_code)]

use std::fmt::Display;

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

impl<T: Ord + Eq + Copy + Display> NumericColumnSketch<T> {
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
  /// Step 2:
  ///
  pub fn construct(input: Vec<T>) -> NumericColumnSketch<T> {
    let _n = input.len();
    let mut input = input;
    input.sort();

    let _frequent_values = Self::generate_frequent_values(&input, COMPRESSION_MAP_SIZE);

    unimplemented!()
  }

  /// Scan over the input data and generate all eligible frequent values
  ///
  /// * `input`: Sorted input values
  /// * `c`: Number of bucket/code in the compression map
  pub(self) fn generate_frequent_values(input: &[T], c: usize) -> Vec<FrequentValueEntry> {
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
    (c * midpoint) / n
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

  use rstest::rstest;

  use crate::NumericColumnSketch;

  #[rstest]
  #[case(1000, 256)]
  #[case(10000, 256)]
  #[case(20000, 256)]
  #[case(20000, 512)]
  fn test_frequent_values_sanity(#[case] dataset_size: usize, #[case] map_size: usize) {
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    // frequent_occurrence = Math.floor(n / c)
    let frequent_occurrence = dataset_size / map_size + 1;

    let mut rng = SmallRng::seed_from_u64(64);

    let mut data = Vec::new();
    let mut current_value = 0;
    let mut is_frequent = BTreeSet::new();
    let mut last_frequent_code = 0;

    while data.len() < dataset_size {
      let current_start = data.len();

      let current_code_if_frequent =
        map_size * ((current_start + current_start + frequent_occurrence) / 2) / dataset_size;

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
}
