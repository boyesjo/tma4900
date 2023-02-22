# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HORIZON = 250_000
N_SIMULATIONS = 100
P_LIST = np.array([0.5, 0.505])
DELTA = 0.01
C1 = 2

# %%
df_list = []
for i in range(100):
    try:
        df = pd.read_csv(f"qucb1_{i}.csv")
        df["simulation"] = i
        df.regret = df.regret.cumsum()
        df = df[df.turn < HORIZON]
        df_list.append(df)
    except FileNotFoundError:
        print(f"file qucb1_{i}.csv not found")
        pass

df = pd.concat(df_list).set_index(["simulation", "turn"])


# %%
def ucb(p_list: np.ndarray, horizon: float, delta: float = 0.1):
    # regular ucb1 alg for comparison
    n_arms = len(p_list)
    est_list = np.zeros_like(p_list)
    times_pulled = np.zeros_like(p_list)
    reg = []

    for arm in range(n_arms):
        est_list[arm] = np.random.binomial(1, p_list[arm])
        times_pulled[arm] += 1
        reg.append(max(p_list) - p_list[arm])

    for t in range(int(horizon) - n_arms):
        arm = np.argmax(est_list + np.sqrt(2 * np.log(t) / times_pulled))
        reg.append(max(p_list) - p_list[arm])

        reward = np.random.binomial(1, p_list[arm])
        est_list[arm] = (est_list[arm] * times_pulled[arm] + reward) / (
            times_pulled[arm] + 1
        )

        times_pulled[arm] += 1

    return np.asarray(reg)


df_ucb = pd.concat(
    [
        pd.DataFrame(
            {
                "turn": np.arange(HORIZON),
                "regret": np.cumsum(ucb(P_LIST, HORIZON)),
                "simulation": i,
            }
        )
        for i in range(N_SIMULATIONS)
    ]
).set_index(["simulation", "turn"])


# %%
df_ucb.to_csv("ucb.csv")

# %%
df.regret.groupby("turn").mean().plot()
plt.fill_between(
    df.regret.groupby("turn").mean().index,
    df.regret.groupby("turn").min(),
    df.regret.groupby("turn").max(),
    alpha=0.2,
)

df_ucb.regret.groupby("turn").mean().plot()
plt.fill_between(
    df_ucb.regret.groupby("turn").mean().index,
    df_ucb.regret.groupby("turn").min(),
    df_ucb.regret.groupby("turn").max(),
    alpha=0.2,
)
plt.ylabel("regret")
plt.savefig("regret.png", dpi=300)

# %%
