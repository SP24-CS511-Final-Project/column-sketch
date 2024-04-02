import pandas as pd

df = pd.DataFrame({"col": range(100)})
block = df[10: 20]
print(block)