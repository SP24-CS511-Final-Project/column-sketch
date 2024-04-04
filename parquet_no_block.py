import pandas as pd
import pyarrow as pa
import random
import pyarrow.parquet as pq
import math
import time

def get_sortedness(df: pd.DataFrame) -> float:
    N = len(df)
    asc = desc = eq = 0
    for i in range(N - 1):
        if df.iloc[i]["col"] < df.iloc[i + 1]["col"]:
            asc += 1
        elif df.iloc[i]["col"] == df.iloc[i + 1]["col"]:
            eq += 1
        else:
            desc += 1
    sortedness = ((max(asc, desc) + eq - math.floor(N / 2))
                  / (math.ceil(N / 2) - 1))
    return sortedness

def degrade_sortedness_to_target(df: pd.DataFrame, target: float) -> None:
    N = len(df)
    index_values = df.index.tolist()
    # prevent infinite loop in case target can't be reach
    for _ in range(100000):
        sortedness = get_sortedness(df)
        if sortedness <= target:
            return
        idx1 = random.choice(index_values)
        idx2 = random.choice(index_values)
        tmp = df['col'][idx1].copy()
        df.at[idx1, 'col'] = df.at[idx2, 'col']
        df.at[idx2, 'col'] = tmp

# define hyperparameters
target_sortedness = [0.2, 0.4, 0.6, 0.8, 1.0]
size_table = 4096

print(f"table size = {size_table}\n")
for target in target_sortedness:
    start_time = time.time()
    print(f"target sortedness = {target}")
    float_series = pd.Series(range(size_table), dtype=float)
    df = pd.DataFrame(float_series, columns=['col'])
    degrade_sortedness_to_target(df, target)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, f"pqs/sortedness={target}.parquet")
    sortedness = get_sortedness(df)
    print(f"real sortedness = {sortedness}")
    end_time = time.time()
    print(f"time spent = {end_time - start_time} \n")
