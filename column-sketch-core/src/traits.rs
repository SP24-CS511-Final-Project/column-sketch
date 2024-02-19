/// The [`Numeric`] trait is a minimal trait describing a numeric data type that we could
/// construct [`NumericColumnSketch`] on.
pub trait Numeric: sealed::Sealed + Copy + PartialEq + PartialOrd {
  /// The last bucket of the compression map always contains the max value of
  /// the data type so that any value greater than the max sample value has a code to fall
  /// into.
  fn max_value() -> Self;
}

impl Numeric for f64 {
  fn max_value() -> Self {
    f64::MAX
  }
}

impl Numeric for u64 {
  fn max_value() -> Self {
    u64::MAX
  }
}

impl Numeric for i64 {
  fn max_value() -> Self {
    i64::MAX
  }
}

impl Numeric for i32 {
  fn max_value() -> Self {
    i32::MAX
  }
}

mod sealed {
  pub trait Sealed {}

  impl Sealed for f64 {}
  impl Sealed for u64 {}
  impl Sealed for i64 {}
  impl Sealed for i32 {}
}
