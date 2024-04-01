import pandas as pd
import pyarrow as pa
import random
import pyarrow.parquet as pq
import math

def print_pq(table_name: str) -> None:
    table = pq.read_table(table_name)
    df = table.to_pandas()
    print(df)

def get_sortedness_block(block: pd.DataFrame) -> float:
    N = len(block)
    asc = desc = eq = 0
    for i in range(N - 1):
        if block.iloc[i]["col"] < block.iloc[i + 1]["col"]:
            asc += 1
        elif block.iloc[i]["col"] == block.iloc[i + 1]["col"]:
            eq += 1
        else:
            desc += 1
    sortedness = ((max(asc, desc) + eq - math.floor(N / 2))
                  / (math.ceil(N / 2) - 1))
    return sortedness

def get_sortedness(df: pd.DataFrame, size_block=512) -> float:
    num_rows = len(df)
    num_full_blocks = num_rows // size_block

    # sum each full block's sortedness
    idx_block_head = 0  # inclusive
    sum_sortedness = 0
    for _ in range(num_full_blocks):
        idx_block_tail = idx_block_head + size_block    # exclusive
        block = df[idx_block_head: idx_block_tail]
        sum_sortedness += get_sortedness_block(block)
        idx_block_head = idx_block_tail

    # if all blocks are full
    if num_rows % size_block == 0:
        return sum_sortedness / num_full_blocks
    else:
        block = df[idx_block_head: num_rows]
        sum_sortedness += get_sortedness_block(block)
        return sum_sortedness / (num_full_blocks + 1)

df = pd.DataFrame({"col": [random.uniform(0, 100) for _ in range(1000)]})
table = pa.Table.from_pandas(df)
pq.write_table(table, "example.parquet")
