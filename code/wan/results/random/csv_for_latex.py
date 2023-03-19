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
    df_all[f"{name}_mean"] = df.groupby("turn").regret.mean()

df_all.to_csv(FOLDER / "plot_data.csv")

# %%
# same but turns spaced on log scale
turns = np.geomspace(1, HORIZON - 1, 200).astype(int)

# if duplicate turns, set to higest not in list
for i in range(1, len(turns)):
    if turns[i] in turns[:i]:
        turns[i] = turns[i - 1] + 1
turns

df_all = pd.DataFrame()
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    df = df.loc[turns]
    df_all[f"{name}_mean"] = df.groupby("turn").regret.mean()

df_all.to_csv(FOLDER / "plot_data_log.csv")

# %%
