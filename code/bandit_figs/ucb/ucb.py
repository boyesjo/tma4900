# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
rs = np.random.RandomState(1337)

HORIZON = 10000
P_LIST = np.array([0.2, 0.4, 0.7])
N_ARMS = len(P_LIST)

estimates = np.zeros((HORIZON, N_ARMS))
times_pulled = np.zeros((HORIZON, N_ARMS), dtype=int)
upper_bounds = np.full((HORIZON, N_ARMS), np.inf)

for t in range(N_ARMS):
    arm = t
    reward = rs.binomial(1, P_LIST[arm])
    if t > 0:
        estimates[t] = estimates[t - 1]
        upper_bounds[t] = upper_bounds[t - 1]
        times_pulled[t] = times_pulled[t - 1]

    times_pulled[t, arm] += 1
    estimates[t, arm] = reward
    upper_bounds[t, arm] = np.sqrt(2 * np.log(t + 1) / 1)

for t in range(N_ARMS, HORIZON):

    arm = np.argmax(estimates[t - 1] + upper_bounds[t - 1])
    reward = rs.binomial(1, P_LIST[arm])

    times_pulled[t] = times_pulled[t - 1]
    times_pulled[t, arm] += 1

    upper_bounds[t] = np.sqrt(2 * np.log(t) / times_pulled[t])
    estimates[t] = estimates[t - 1]
    estimates[t, arm] = (estimates[t - 1, arm] * (t - 1) + reward) / t

    print(estimates[t])
    print(upper_bounds[t])
    print(times_pulled[t])
    print()


# %%
def plot(turn: int) -> None:
    t = turn - 1
    # plot true means, estimates and error bars
    plt.errorbar(
        x=np.arange(N_ARMS),
        y=estimates[t],
        yerr=upper_bounds[t],
        fmt="o",
    )
    # plot horizontal lines at true means
    plt.hlines(
        y=P_LIST,
        xmin=-0.2 + np.arange(N_ARMS),
        xmax=0.2 + np.arange(N_ARMS),
        linestyles="dashed",
        color="black",
    )
    plt.title(f"Upper Confidence Bound, turn {turn}")
    plt.xlabel("Arm")
    plt.xticks(np.arange(N_ARMS))
    plt.ylabel("Estimate")
    # add number of pulls for each arm
    for i in range(N_ARMS):
        plt.text(
            x=i - 0.1,
            y=estimates[t, i] + upper_bounds[t, i] + 0.01,
            s=f"{times_pulled[t, i]}",
        )
    plt.show()


# %%
# plot(10)
# plot(100)
# plot(1000)
# plot(10000)

# %%
# estimates, counts and ucb for turns 10, 100, 1000, 10000
turns = [10, 100, 1000, 10000]
for turn in turns:
    pd.DataFrame(
        {
            "arm": np.arange(N_ARMS) + 1,
            "estimate": estimates[turn - 1],
            "ucb": upper_bounds[turn - 1],
            "count": times_pulled[turn - 1],
        }
    ).to_csv(f"ucb_{turn}.csv", index=False)
