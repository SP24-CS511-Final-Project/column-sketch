//! This crate consists of the integration point between column sketch and apache parquet. It has responsibilities for:
//! 1. Provide infrastructure for building column sketch on a parquet table and output a transformed table
//! 2. Provide a binary for performing the transformation based on command-line arguments
//! 3. Implement infrastructure for performing predicated scan on an end-to-end parquet reading workflow
//!     3.1 We should search for optimal combination of configurations for
//!         (a). Plain untransformed parquet table with zone map
//!         (b). Plain untransfored parquet table with column index + zone map
//!         (c). Parquet table with column sketch
//!     3.2 Possible configuration space includes encodings & compressions of the data column, encoding & compressions of the code column, serialization scheme of compression map, etc.

#[test]
fn it_works() {
  use arrow::array::Int32Array;
  let _array = Int32Array::from(vec![Some(1), None, Some(3)]);
}
