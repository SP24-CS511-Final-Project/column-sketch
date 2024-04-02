import pandas as pd
import pyarrow as pa
import random
import pyarrow.parquet as pq
import math

def print_pq(table_name: str) -> None:
    table = pq.read_table(table_name)
    df = table.to_pandas()
    for i in range(len(df)):
        print(df["col"][i])

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

def get_sortedness(df: pd.DataFrame) -> float:
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

def degrade_sortedness_to_target(df: pd.DataFrame) -> None:
    # swap random value pairs in the block till reach target_sortedness
    def degrade_block(block: pd.DataFrame) -> None:
        N = len(block)
        index_values = block.index.tolist()
        # prevent infinite loop in case target can't be reach
        for _ in range(10000):
            sortedness_block = get_sortedness_block(block)
            if sortedness_block <= target_sortedness:
                return
            idx1 = random.choice(index_values)
            idx2 = random.choice(index_values)
            tmp = block['col'][idx1].copy()
            block.at[idx1, 'col'] = block.at[idx2, 'col']
            block.at[idx2, 'col'] = tmp

    num_rows = len(df)
    num_full_blocks = num_rows // size_block

    # degrade each block's sortedness to target
    idx_block_head = 0  # inclusive
    for _ in range(num_full_blocks):
        idx_block_tail = idx_block_head + size_block    # exclusive
        block = df[idx_block_head: idx_block_tail]
        degrade_block(block)
        idx_block_head = idx_block_tail

    if num_rows % size_block != 0:
        block = df[idx_block_head: num_rows]
        degrade_block(block)

# target_sortedness = float(input("Enter the target sortedness: "))
# size_block = int(input("Enter the block size: "))
target_sortedness = 0.8
size_block = 512

float_series = pd.Series(range(1000), dtype=float)
df = pd.DataFrame(float_series, columns=['col'])
degrade_sortedness_to_target(df)
table = pa.Table.from_pandas(df)
pq.write_table(table, "example.parquet")
# print(f"Sortedness of table generated = {get_sortedness(df)}")
# print_pq("example.parquet")
