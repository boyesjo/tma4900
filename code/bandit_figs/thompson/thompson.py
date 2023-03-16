# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from scipy.stats import beta

# %%
rs = np.random.RandomState(1337)

HORIZON = 10000
P_LIST = np.array([0.2, 0.4, 0.7])
N_ARMS = len(P_LIST)

posteriors = [[{"a": 1, "b": 1} for _ in range(N_ARMS)]]

for t in range(1, HORIZON):
    arm = np.argmax([beta.rvs(**posterior) for posterior in posteriors[t - 1]])
    reward = rs.binomial(1, P_LIST[arm])

    posteriors.append(copy.deepcopy(posteriors[t - 1]))
    posteriors[t][arm]["a"] += reward
    posteriors[t][arm]["b"] += 1 - reward

    print(posteriors[t])


# %%
def plot(turn: int) -> None:
    t = turn - 1
    # plot posteriors for each arm in each subplot
    for arm in range(N_ARMS):
        plt.subplot(1, N_ARMS, arm + 1)
        x = np.linspace(0, 1, 100)
        plt.plot(x, beta.pdf(x, **posteriors[t][arm]))
    plt.show()


plot(1)
plot(10)
plot(100)
plot(1000)


# %%
# posteriors as csvs
for t in (0, 10, 100, 1000):
    x = np.linspace(0, 1, 100)
    df = pd.DataFrame(
        {
            "x": x,
            "y1": beta.pdf(x, **posteriors[t][0]),
            "y2": beta.pdf(x, **posteriors[t][1]),
            "y3": beta.pdf(x, **posteriors[t][2]),
        }
    )
    df.to_csv(f"thompson_{t}.csv", index=False)
