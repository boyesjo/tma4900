# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P_LIST = np.array([0.5, 0.51])
HORIZON = 100000
DELTA = 0.01
N_SIMULATIONS = 100
C2 = 2

# %%
FREQ = 1
df_qucb = pd.read_parquet("qucb.parquet").set_index(["simulation", "turn"])
df_qucb = df_qucb[df_qucb.index.get_level_values("turn") % FREQ == 0]
# df_ucb = pd.read_parquet("ucb.parquet").set_index(["simulation", "turn"])
# df_ucb = df_ucb[df_ucb.index.get_level_values("turn") % FREQ == 0]
# df_thomp = pd.read_parquet("thompson.parquet").set_index(
#     ["simulation", "turn"]
# )
# df_thomp = df_thomp[df_thomp.index.get_level_values("turn") % FREQ == 0]


# %%
def plot(df, color, label, error="range"):
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
# plot(df_ucb, "red", "UCB")
# plot(df_thomp, "green", "Thompson")
plt.legend()
# plt.savefig("regret.png", dpi=300)


# %%
df_list = []
for i in range(100):
    df = pd.read_parquet(f"ucb_{i}.parquet")
    df.regret = df.regret.cumsum()
    df["turn"] = np.arange(len(df))
    df["simulation"] = i
    df_list.append(df)
df = pd.concat(df_list).set_index(["simulation", "turn"])
df
# %%
df_qucb.reset_index().to_parquet(
    "qucb.parquet", index=False, compression="brotli"
)
# %%
