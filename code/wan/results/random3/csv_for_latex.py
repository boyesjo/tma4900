# %%
import pandas as pd
from pathlib import Path
import numpy as np


FOLDER = Path()

# %%
df_qucb = pd.read_parquet(FOLDER / "qucb.parquet").set_index("turn")
df_ucb = pd.read_parquet(FOLDER / "ucb.parquet").set_index("turn")
df_thomp = (
    pd.read_parquet(FOLDER / "thomp.parquet")
    .set_index("turn")
    .rename({"thompson": "thomp"}, axis=1)
)

# %%
HORIZON = df_qucb.index.max()  # + 1

# %%
# print number of simulations for each df
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    n_sims = df["sim"].nunique()
    print(f"{name}: {n_sims}")

# %%
turns = np.linspace(1, HORIZON - 1, 200).astype(int)
# %%
df_all = pd.DataFrame()
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    # grab 200 turns evenly spaced
    df.sort_index(inplace=True)
    df = df.loc[turns]
    df_all[f"{name}_mean"] = df.groupby("turn")[name].mean()

# %%
df_all.to_csv(FOLDER / "plot_data.dat")

# %%
