# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_list = []
for i in range(1, 100):
    try:
        df = pd.read_csv(f"big1/qucb1_{i}.csv")
        df["i"] = i
        df_list.append(df)
        df.regret = df.regret.cumsum()
    except FileNotFoundError:
        pass

df = pd.concat(df_list).set_index(["turn", "i"])
# %%
df.regret.groupby("turn").mean().plot(xlim=(0, 1e5))

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


# %%
# run ucb for comparison, 1e5 turns, 50 times
# p_list = np.array([0.5, 0.501])
# reg_list = []
# for i in range(50):
#     reg_list.append(ucb(p_list, 1e5))

# %%
# df_ucb = pd.DataFrame(reg_list).T
# mean and plot
# %%
# plot with mean and std
df.regret.groupby("turn").mean().iloc[: int(1e5)].plot()
plt.fill_between(
    df.regret.groupby("turn").mean().iloc[: int(1e5)].index,
    df.regret.groupby("turn").quantile(0.10).iloc[: int(1e5)].values,
    df.regret.groupby("turn").quantile(0.90).iloc[: int(1e5)].values,
    alpha=0.2,
)
df.regret.groupby("turn").mean().iloc[: int(1e5)].plot()
plt.fill_between(
    df.regret.groupby("turn").mean().iloc[: int(1e5)].index,
    df.regret.groupby("turn").quantile(0.01).iloc[: int(1e5)].values,
    df.regret.groupby("turn").quantile(0.99).iloc[: int(1e5)].values,
    alpha=0.1,
)
# std

# df_ucb.cumsum(axis=0).mean(axis=1).plot()
# plt.fill_between(
#     df_ucb.cumsum(axis=0).mean(axis=1).index,
#     df_ucb.cumsum(axis=0).mean(axis=1).values
#     - df_ucb.cumsum(axis=0).std(axis=1).values,
#     df_ucb.cumsum(axis=0).mean(axis=1).values
#     + df_ucb.cumsum(axis=0).std(axis=1).values,
#     alpha=0.2,
# )
# %%
