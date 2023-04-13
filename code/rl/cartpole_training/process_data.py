# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILES = {
    "DQN": "DQN.csv",
    "PPO": "PPO.csv",
    "A2C": "A2C.csv",
    "QNN": "qreinforce.csv",
}


def read_csv(file, name):
    return (
        pd.read_csv(file, index_col=0, header=None)
        .T.iloc[:, 1:]
        .reset_index(names=["episode"])
        .melt(
            id_vars=["episode"],
            var_name="simulation",
            value_name=name,
        )
        .assign(simulation=lambda x: x.simulation.astype(int))
        .set_index(["simulation", "episode"])
    )


df = pd.concat(
    [read_csv(file, name) for name, file in FILES.items()], axis=1
).reset_index()
df
# %%
# get rolling mean within each simulation then get mean of all simulations
df_proc = (
    df.groupby("simulation")
    .rolling(window=20, min_periods=1, center=False, on="episode")
    .mean()
    .groupby("episode")
    .mean()
)

df_proc.plot()
# %%
df_proc.to_csv("results.dat")

# save only every 10th episode
df_proc.iloc[0::10].to_csv("results_10.dat")
# %%
