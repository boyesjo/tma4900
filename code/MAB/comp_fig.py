# %%
import numpy as np
import agents
import matplotlib.pyplot as plt
from arms.bernoulli import BernoulliArm
from bandits import Bandits
import pandas as pd

N_TURNS = 1000
N_SIMS = 1000
P_LIST = np.array([0.2, 0.8])
N_ARMS = P_LIST.shape[0]
bandits = Bandits([BernoulliArm(p) for p in P_LIST])


agent_list: list[agents.Agent] = [
    agents.Random,
    agents.Greedy,
    agents.EpsilonGreedy,
    agents.UCB,
    agents.ThompsonBernoulli,
]

regrets = np.zeros((len(agent_list), N_TURNS, N_SIMS))
for sim in range(N_SIMS):
    for i, agent_constructor in enumerate(agent_list):
        agent = agent_constructor(bandits)
        agent.play(N_TURNS)
        regrets[i, :, sim] = agent.regret


# %%
plt.plot(regrets.mean(axis=2).T)
plt.legend([agent.__name__ for agent in agent_list])
# %%

# save means to csv
df = pd.DataFrame(
    {
        agent.__name__: regrets.mean(axis=2).T[:, i]
        for i, agent in enumerate(agent_list)
    }
)
# start index at 1
df.index += 1
df.to_csv("comparison.dat", index_label="turn")
# %%
