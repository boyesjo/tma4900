# %%
import pandas as pd
from pathlib import Path
import numpy as np


FOLDER = Path()

# %%
df_qucb = pd.read_parquet(FOLDER / "qucb.parquet").set_index("turn")
df_ucb = pd.read_parquet(FOLDER / "ucb.parquet").set_index("turn")
df_thomp = pd.read_parquet(FOLDER / "thomp.parquet").set_index("turn")

HORIZON = df_qucb.index.max() + 1

# %%
df_all = pd.DataFrame()
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    # grab 200 turns evenly spaced
    df = df.iloc[:: int(HORIZON / 200)]
    df_all[f"{name}_mean"] = df.groupby("turn")[
        name if name != "thomp" else "thompson"
    ].mean()

df_all.to_csv(FOLDER / "plot_data.csv")

# %%

# %% open all and rename column to regret and save again
