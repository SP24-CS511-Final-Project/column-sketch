import pandas as pd
import pyarrow as pa
import random
import pyarrow.parquet as pq

def print_pq(table_name: str) -> None:
    table = pq.read_table(table_name)
    df = table.to_pandas()
    print(df)

df = pd.DataFrame({"col": [random.uniform(0, 100) for _ in range(1000)]})
table = pa.Table.from_pandas(df)
pq.write_table(table, "example.parquet")


