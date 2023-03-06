# %%
from pathlib import Path

import pandas as pd

folder = "big1"
prefix = "thompson"

# %%
df_list = []
for i in range(100):
    df = pd.read_csv(Path() / folder / f"{prefix}_{i}.csv")
    df["turn"] = df.index
    df["sim"] = i
    df.set_index(["turn", "sim"], inplace=True)
    df_list.append(df)

df = pd.concat(df_list)
# %%
df.reset_index().to_parquet(
    Path() / folder / f"{prefix}.parquet", index=False, compression="brotli"
)
