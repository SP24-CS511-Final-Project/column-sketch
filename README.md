# column-sketch

This is the main repository of the final project of CS511 offered at Spring 2024 semester. 

## Project Goals

The goal of the project is to reproduce [column sketch](https://stratos.seas.harvard.edu/files/stratos/files/sketches.pdf) on top of [apache parquet](https://parquet.apache.org/) format, 
and to explore how and if it improves performance on the workload of scanning a parquet table with a predicate to evaluate and produce a position list of rows that satisfy the predicate.

### Column Sketch

The essential idea of column sketch is that given an array of data (either numerical or string, but we will prototype numerical data first), we will output a compression map, which is an order-preserving map from the original data to 8-bit integer space.

Assume the original dataset is denoted by $D$, the compression map is a function denoted by $S$, then by order-preserving, we mean the following:

$\forall a, b \in D, a >= b \rightarrow S(a) \geq S(b)$

And the useful corollary is that $S(a) < S(b) \rightarrow a < b$, which will be used for evaluating range queries on the compressed column.

Concretely, we want to implement something like follows:

```Rust
pub trait CompressionMap<T> {
  fn compress(value: T) -> u8
}

fn column_sketch<T>(data: Vec<T>) -> impl CompressionMap<T> {}
```

### Transform Parquet Tables

We will also implement a tool on top of the core algorithm, which automatically implement the compression for a parquet table, which involves transforming the table and serializing the compression map.

- We store compressed columns directly as additional UINT8 columns in the parquet.
- As a first step, we will serialize the compression map into a separate file so that we need not change parquet internals. It is also possible to store the serialized compression map directly as a distinct parquet page.
