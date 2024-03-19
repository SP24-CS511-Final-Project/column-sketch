use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::{
  array::{ArrayRef, Int32Array},
  datatypes::{DataType, Field, Schema},
  record_batch::RecordBatch,
};
use clap::Parser;
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};

/// Reads a Parquet file, adds a column, and writes to a new Parquet file.
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
  /// Sets the input Parquet file to use
  input: String,
  #[clap(long, help = "Name of the column to add sketch on")]
  column: String,
  /// Sets the output Parquet file to write
  #[clap(long, short, help = "Output parquet file path")]
  output: String,
}

fn main() -> Result<()> {
  let args = Args::parse();

  let file = File::open(Path::new(&args.input))?;
  let reader = ParquetRecordBatchReaderBuilder::try_new(file)?
    .with_batch_size(1024)
    .build()?;
  let mut output_file = File::create(Path::new(&args.output))?;

  // Assuming all batches have the same schema, let's process only the first one for simplicity
  for batch in reader {
    let batch = batch?;
    // Create new column
    let new_column_len = batch.num_rows();
    let new_column = Int32Array::from(vec![42; new_column_len]);
    let new_column: ArrayRef = Arc::new(new_column);

    // Add new column to existing batch
    let schema = batch.schema();
    let new_field = Arc::new(Field::new("new_int_column", DataType::Int32, false));
    let new_schema = Schema::new(
      schema
        .fields()
        .iter()
        .cloned()
        .chain(std::iter::once(new_field))
        .collect::<Vec<_>>(),
    );

    let new_batch = RecordBatch::try_new(
      Arc::new(new_schema),
      batch
        .columns()
        .iter()
        .cloned()
        .chain(std::iter::once(new_column))
        .collect(),
    )?;

    // Write new batch to a new Parquet file
    let mut arrow_writer = ArrowWriter::try_new(&mut output_file, new_batch.schema(), None)?;

    arrow_writer.write(&new_batch)?;
  }

  Ok(())
}
