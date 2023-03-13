# %%
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical import lai_robbins_bound, ucb_bound

FOLDER = Path("results") / "low_prob_fix2"

# %%
df_qucb = pd.read_parquet(FOLDER / "qucb.parquet")
df_ucb = pd.read_parquet(FOLDER / "ucb.parquet")
df_thomp = pd.read_parquet(FOLDER / "thompson.parquet")
with open(FOLDER / "settings.pickle", "rb") as f:
    settings = pickle.load(f)


# %%
def plot(df, color, name, err):
    d = df.groupby("turn").regret.mean()
    plt.plot(d.index, d, color=color, label=name)
    if err == "std":
        plt.fill_between(
            d.index,
            d - df.groupby("turn").regret.std(),
            d + df.groupby("turn").regret.std(),
            color=color,
            alpha=0.2,
        )
    elif err == "range":
        plt.fill_between(
            d.index,
            df.groupby("turn").regret.min(),
            df.groupby("turn").regret.max(),
            color=color,
            alpha=0.2,
        )
    elif isinstance(err, float):
        plt.fill_between(
            d.index,
            df.groupby("turn").regret.quantile(err),
            df.groupby("turn").regret.quantile(1 - err),
            color=color,
            alpha=0.2,
        )


# %%
ERR = "std"
plot(df_qucb, "red", "QUCB1", ERR)
plot(df_thomp, "blue", "Thompson", ERR)
plot(df_ucb, "green", "UCB1", ERR)
plt.legend()
delta = np.abs(settings["p_list"][1] - settings["p_list"][0])
plt.title(f"{settings['p_list']}, $\\Delta$ = {delta}")
plt.savefig(FOLDER / "plot.png", dpi=300)

# plt.plot(
#     lai_robbins_bound(settings["p_list"], settings["horizon"]),
#     "k--",
#     label="Lai-Robins",
# )

# plt.plot(
#     ucb_bound(settings["p_list"], settings["horizon"]),
#     "k--",
#     label="UCB1 bound",
# )

# %%
