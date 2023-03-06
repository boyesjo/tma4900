# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 250_000
N_SIMULATIONS = 100
P_LIST = np.array([0.99, 0.9905])
DELTA = 0.01

# # %%
# df_list = []
# for sim in range(100):
#     df = pd.read_csv(f"qucb_{sim}.csv")
#     df["turn"] = np.arange(HORIZON)
#     df["sim"] = sim
#     df_list.append(df)
# df_qucb = pd.concat(df_list)
# # %%
# df_list = []
# for sim in range(28):
#     df = pd.read_csv(f"ucb_{sim}.csv")
#     df["turn"] = np.arange(HORIZON)
#     df["sim"] = sim
#     df_list.append(df)
# df_ucb = pd.concat(df_list)
# # %%
# df_list = []
# for sim in range(28):
#     df = pd.read_csv(f"thompson_{sim}.csv")
#     df["turn"] = np.arange(HORIZON)
#     df["sim"] = sim
#     df_list.append(df)
# df_thompson = pd.concat(df_list)

df_qucb = pd.read_parquet("qucb.parquet").set_index(["turn", "sim"])
df_ucb = pd.read_parquet("ucb.parquet").set_index(["turn", "sim"])
df_thompson = pd.read_parquet("thompson.parquet").set_index(["turn", "sim"])

# %%
def plot(df, color, label, err="std"):
    d = df.groupby("turn").mean()
    plt.plot(d.index, d["regret"], color=color, label=label)

    if err == "std":
        plt.fill_between(
            d.index,
            d["regret"] - df.groupby("turn").std()["regret"],
            d["regret"] + df.groupby("turn").std()["regret"],
            color=color,
            alpha=0.2,
        )
    elif err == "range":
        plt.fill_between(
            d.index,
            d["regret"] - df.groupby("turn").min()["regret"],
            d["regret"] + df.groupby("turn").max()["regret"],
            color=color,
            alpha=0.2,
        )
    elif isinstance(err, float):
        plt.fill_between(
            d.index,
            df.groupby("turn").quantile(err)["regret"],
            df.groupby("turn").quantile(1 - err)["regret"],
            color=color,
            alpha=0.2,
        )


err = "std"
plot(df_qucb, "blue", "QUCB1", err=err)
plot(df_ucb, "red", "UCB1", err=err)
plot(df_thompson, "green", "Thompson", err=err)

# %%
