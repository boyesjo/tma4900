# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
df_list = []
for sim in range(100):
    df = pd.read_csv(f"results/results_{sim}.csv")
    df["sim"] = sim
    df_list.append(df)
df = pd.concat(df_list)

# %%
df["p_delta"] = df["p1"] - df["p2"]
df["logit_delta"] = np.log(df["p1"] / (1 - df["p1"])) - np.log(
    df["p2"] / (1 - df["p2"])
)

df["regret_ratio"] = df["thomp_regret"] / df["ucb_regret"]


# %%
def plot(y):
    x = "p_delta"
    mean = df.groupby(x)[y].mean()
    mean.plot(label=y)
    plt.fill_between(
        mean.index,
        mean - df.groupby(x)[y].std(),
        mean + df.groupby(x)[y].std(),
        alpha=0.2,
    )


plot("ucb_regret")
plot("thomp_regret")
plt.xscale("log")
plt.legend()
plt.show()

plot("regret_ratio")
plt.ylabel("Thompson / UCB")
plt.xscale("log")
plt.show()

# %%
plt.scatter(df["logit_delta"], df["regret_ratio"])
plt.xscale("log")
# %%
