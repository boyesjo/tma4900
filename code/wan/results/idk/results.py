# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 250_000
N_SIMULATIONS = 100
P_LIST = np.array([0.05, 0.01])
DELTA = 0.01


# %%
df_qucb = pd.read_parquet("qucb.parquet").set_index(["sim", "turn"])
df_thompson = pd.read_parquet("thompson.parquet").set_index(["sim", "turn"])
df_ucb = pd.read_parquet("ucb.parquet").set_index(["sim", "turn"])


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


ERR = "std"
plot(df_qucb, "red", "QUCB1", ERR)
plot(df_thompson, "blue", "Thompson", ERR)
plot(df_ucb, "green", "UCB", ERR)
plt.legend()

# %%
