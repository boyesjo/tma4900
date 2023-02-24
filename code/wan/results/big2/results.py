# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 250_000
N_SIMULATIONS = 100
P_LIST = np.array([0.5, 0.505])
DELTA = 0.01
C1 = 2

df_qucb = pd.read_parquet("qucb.parquet").set_index(["simulation", "turn"])
df_qucb = df_qucb[df_qucb.index.get_level_values("turn") % 1_000 == 0]
df_ucb = pd.read_parquet("ucb.parquet").set_index(["simulation", "turn"])
df_ucb = df_ucb[df_ucb.index.get_level_values("turn") % 1_000 == 0]


# %%
def plot(df, color, label):
    # df.regret.groupby("turn").median().plot(color=color)
    df.regret.groupby("turn").mean().plot(color=color)
    # plt.fill_between(
    #     df.regret.groupby("turn").mean().index,
    #     df.regret.groupby("turn").quantile(0.05),
    #     df.regret.groupby("turn").quantile(0.95),
    #     alpha=0.3,
    #     color=color,
    # )
    # plt.fill_between(
    #     df.regret.groupby("turn").mean().index,
    #     df.regret.groupby("turn").min(),
    #     df.regret.groupby("turn").max(),
    #     alpha=0.1,
    #     color=color,
    # )


plot(df_qucb, "blue", "QUCB")
plot(df_ucb, "red", "UCB")


# plt.savefig("regret.png", dpi=300)


# %%