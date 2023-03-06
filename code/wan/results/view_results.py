# %%
import matplotlib.pyplot as plt
import pandas as pd

FOLDER = "big1"

# %%
df_qucb = pd.read_parquet(FOLDER + "/qucb.parquet")
df_ucb = pd.read_parquet(FOLDER + "/ucb.parquet")
df_thomp = pd.read_parquet(FOLDER + "/thompson.parquet")


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
plot(df_ucb, "green", "UCB", ERR)
plt.legend()
