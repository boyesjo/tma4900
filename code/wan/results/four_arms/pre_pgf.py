# %%
from pathlib import Path

import numpy as np
import pandas as pd

FOLDER = Path()

# %%
df_qucb = pd.read_parquet(FOLDER / "qucb.parquet").set_index("turn")
df_ucb = pd.read_parquet(FOLDER / "ucb.parquet").set_index("turn")
df_thomp = pd.read_parquet(FOLDER / "thompson.parquet").set_index("turn")

HORIZON = df_qucb.index.max() + 1


# %%
def std_err(df):
    mean = df.groupby("turn").regret.mean()
    std = df.groupby("turn").regret.std()
    return mean - std, mean + std


def mean_std_err(df):
    mean = df.groupby("turn").regret.mean()
    std = df.groupby("turn").regret.std() / np.sqrt(
        df.groupby("turn").regret.count()
    )
    return mean - std, mean + std


def range_err(df):
    return df.groupby("turn").regret.min, df.groupby("turn").regret.max()


def quantile_err(df, q):
    low = df.groupby("turn").regret.quantile(q)
    high = df.groupby("turn").regret.quantile(1 - q)
    return low, high


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
    df_all[f"{name}_low"], df_all[f"{name}_high"] = mean_std_err(df)


df_all.to_csv(FOLDER / "plot_data.csv")
# %%
