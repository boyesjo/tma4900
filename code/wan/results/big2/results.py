# %%
import lai_robbins
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 250_000
N_SIMULATIONS = 100
P_LIST = np.array([0.5, 0.505])
DELTA = 0.01
C1 = 2

# %%
FREQ = 1_000
df_qucb = pd.read_parquet("qucb.parquet").set_index(["simulation", "turn"])
df_qucb = df_qucb[df_qucb.index.get_level_values("turn") % FREQ == 0]
df_ucb = pd.read_parquet("ucb.parquet").set_index(["simulation", "turn"])
df_ucb = df_ucb[df_ucb.index.get_level_values("turn") % FREQ == 0]
df_thomp = pd.read_parquet("thompson.parquet").set_index(
    ["simulation", "turn"]
)
df_thomp = df_thomp[df_thomp.index.get_level_values("turn") % FREQ == 0]


# %%
def plot(df, color, label, error="std"):
    # df.regret.groupby("turn").median().plot(color=color)
    df.regret.groupby("turn").mean().plot(color=color, label=label)
    # plt.xscale("log")
    # plt.yscale("log")
    if isinstance(error, float):
        plt.fill_between(
            df.regret.groupby("turn").mean().index,
            df.regret.groupby("turn").quantile(error),
            df.regret.groupby("turn").quantile(1 - error),
            alpha=0.3,
            color=color,
        )
    elif error == "range":
        plt.fill_between(
            df.regret.groupby("turn").mean().index,
            df.regret.groupby("turn").min(),
            df.regret.groupby("turn").max(),
            alpha=0.1,
            color=color,
        )
    elif error == "std":
        plt.fill_between(
            df.regret.groupby("turn").mean().index,
            df.regret.groupby("turn").mean() - df.regret.groupby("turn").std(),
            df.regret.groupby("turn").mean() + df.regret.groupby("turn").std(),
            alpha=0.3,
            color=color,
        )


plot(df_qucb, "blue", "QUCB")
plot(df_ucb, "red", "UCB")
plot(df_thomp, "green", "Thompson")
plt.plot(
    lai_robbins.bernoulli(P_LIST, HORIZON, step=FREQ),
    "black",
    label="Lai-Robbins",
)

plt.legend()
# plt.savefig("regret.png", dpi=300)


# %%
