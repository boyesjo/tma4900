# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

np.random.seed(1337)

#
# %%
K = 3  # number of arms
T = 1000  # number of time steps

# %%
# true probabilities
p = np.array([0.5, 0.6, 0.3])

posteriors = {i: {"alpha": 1, "beta": 1} for i in range(K)}
times_pulled = {i: 0 for i in range(K)}

rewards = np.zeros(T)


def plot_posteriors(posteriors, title=""):
    plt.figure(figsize=(12, 4))
    for i in range(K):
        plt.subplot(1, K, i + 1)
        x = np.linspace(0, 1, 100)
        y = beta.pdf(x, posteriors[i]["alpha"], posteriors[i]["beta"])
        plt.plot(x, y)
        plt.title(f"arm {i}, pulled {times_pulled[i]} times")
        plt.vlines(p[i], 0, 4, color="red", linestyle="--")
    plt.suptitle(title)
    plt.show()


# %%
plot_posteriors(posteriors, title="initial posteriors")
for t in range(T):
    # sample from the posterior
    samples = {
        i: beta.rvs(posteriors[i]["alpha"], posteriors[i]["beta"])
        for i in range(K)
    }

    # choose the arm with the highest sample
    arm = max(samples, key=samples.get)

    # pull the arm
    reward = np.random.binomial(1, p[arm])

    # update the posterior
    posteriors[arm]["alpha"] += reward
    posteriors[arm]["beta"] += 1 - reward

    # update the total reward
    rewards[t] = reward

    # update the times pulled
    times_pulled[arm] += 1

    if t in [0, 10, 100, 1000]:
        plot_posteriors(posteriors, title=f"posteriors after {t+1} pulls")

# %%
plt.plot(np.linspace(0, 1, 100), beta.pdf(np.linspace(0, 1, 100), 1, 2))
plt.show()
