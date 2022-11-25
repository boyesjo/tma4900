# %%
import agents
import numpy as np
from bandits import BanditsEnv

n_arms = 3
T = 100
N = 20

env = BanditsEnv(n_arms, T)

# %%
agent_dict = {
    "random": agents.Random(env),
    "greedy": agents.Greedy(env),
    "epsilon_greedy": agents.EpsilonGreedy(env, eps=0.1),
    "epsilon_decay": agents.EpsilonDecay(env, eps_0=1),
    "ucb": agents.UCB(env),
    "thompson": agents.ThompsonBernoulli(env),
}

regrets = {name: np.zeros((N, T)) for name in agent_dict.keys()}

# %%
for n in range(N):
    env.reset()
    for name, agent in agent_dict.items():

        obs = env.reset(keep_p_list=True)

        done = False
        while not done:
            action, _ = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
        regrets[name][n] = env.regret

# %%
import matplotlib.pyplot as plt

for name, regret in regrets.items():
    plt.plot(np.mean(regret, axis=0), label=name)
    # add standard deviation
    plt.fill_between(
        np.arange(T),
        np.mean(regret, axis=0) - np.std(regret, axis=0) / (N + 1),
        np.mean(regret, axis=0) + np.std(regret, axis=0) / (N + 1),
        alpha=0.2,
    )

plt.legend()
plt.savefig("regrets_classic.png")
plt.show()
# import matplotlib.pyplot as plt

# print(env.p_list)

# for name, regret in regrets.items():
#     plt.plot(regret, label=name)
# plt.legend()
# plt.yscale("log")
# plt.show()

# %%
import pandas as pd

# create df with regrets for each agent as columns
pd.concat(
    [
        pd.DataFrame(regrets[name])
        .melt(ignore_index=False, var_name="t", value_name=name)
        .set_index("t", append=True)
        for name in regrets.keys()
    ],
    axis=1,
).to_csv("regrets_classic.csv", index_label=["n", "t"])


# %%
