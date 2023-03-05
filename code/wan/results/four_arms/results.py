# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

P_LIST = np.array([0.5, 0.51, 0.52, 0.53])
HORIZON = 250_000
DELTA = 0.01
N_SIMULATIONS = 100


# %%
df_qucb = pd.read_parquet("qucb.parquet").set_index(["simulations", "turn"])
df_ucb = pd.read_parquet("ucb.parquet").set_index(["simulations", "turn"])
df_thompson = pd.read_parquet("thompson.parquet").set_index(
    ["simulations", "turn"]
)


# %%
def plot(df, label, color, error=0.025):
    df_mean = df.groupby("turn").mean()
    df_std = df.groupby("turn").std()
    plt.plot(df_mean.index, df_mean.regret, label=label, color=color)
    if error == "std":
        plt.fill_between(
            df_mean.index,
            df_mean.regret - df_std.regret,
            df_mean.regret + df_std.regret,
            alpha=0.2,
            color=color,
        )
    elif error == "range":
        plt.fill_between(
            df_mean.index,
            df.groupby("turn").min().regret,
            df.groupby("turn").max().regret,
            alpha=0.2,
            color=color,
        )
    elif isinstance(error, float):
        plt.fill_between(
            df_mean.index,
            df.groupby("turn").quantile(1 - error).regret,
            df.groupby("turn").quantile(error).regret,
            alpha=0.2,
            color=color,
        )
    else:
        raise ValueError("error must be 'std', 'range', or a float")


plot(df_ucb, "UCB", "red")
plot(df_qucb, "QUCB", "blue")
plot(df_thompson, "Thompson", "green")
# %%
