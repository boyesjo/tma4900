# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FOLDER = Path("results") / "random"


def load_parquet(name: str) -> pd.DataFrame:
    df = pd.read_parquet(str(FOLDER / f"{name}.parquet"))
    df.set_index(["sim", "turn"], inplace=True)
    return df


# %%
df_qucb = load_parquet("qucb")
df_ucb = load_parquet("ucb")
df_thomp = load_parquet("thomp")

# %% print number of sims for each algorithm
print(len(df_qucb.index.get_level_values("sim").unique()))
print(len(df_ucb.index.get_level_values("sim").unique()))
print(len(df_thomp.index.get_level_values("sim").unique()))


# %%
def load_p_lists():
    return pd.read_parquet(str(FOLDER / "p_lists.parquet")).set_index("sim")


p_lists = load_p_lists()
p_lists


# %%
def plot(df: pd.DataFrame, label: str):
    n_sims = len(df.index.get_level_values("sim").unique())
    d = df.groupby("turn").regret.mean()
    plt.plot(d.index, d, label=label)
    plt.fill_between(
        d.index,
        d - df.groupby("turn").regret.std() / (n_sims**0.5),
        d + df.groupby("turn").regret.std() / (n_sims**0.5),
        alpha=0.2,
    )


plot(df_qucb, "QUCB1")
plot(df_thomp, "Thompson")
plot(df_ucb, "UCB1")
plt.plot(np.arange(250_000) / 6, label="Random", color="black", linestyle="--")
plt.yscale("log")
plt.xscale("log")
plt.legend()


# %%
