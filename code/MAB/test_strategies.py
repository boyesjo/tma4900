# %%
import agents
import matplotlib.pyplot as plt
import numpy as np
from arms.bernoulli import BernoulliArm
from bandits import Bandits

# Create bandits
p_list = np.linspace(0.1, 0.9, 9)

bandits = Bandits([BernoulliArm(p) for p in p_list])

# %%
agent_list = [
    agents.Random(bandits),
    agents.Greedy(bandits, test_rounds=1),
    agents.EpsilonGreedy(bandits, test_rounds=1, eps=0.1),
    agents.EpsilonDecay(bandits, test_rounds=1, eps_0=1),
    agents.UCB(bandits, test_rounds=10),
    agents.ThompsonBernoulli(bandits),
]

# %%
T = 5000

for agent in agent_list:
    for _ in range(T):
        agent.pull()

    plt.plot(agent.regret, label=agent.__class__.__name__)

plt.legend()
plt.show()

# %%
