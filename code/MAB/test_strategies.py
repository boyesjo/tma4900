# %%
import agents
import matplotlib.pyplot as plt
import numpy as np
from arms.bernoulli import BernoulliArm
from bandits import Bandits

# Create bandits
p_list = [0.1, 0.5, 0.45, 0.2]

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
T = 1000

for agent in agent_list:
    for t in range(T):
        agent.pull()

        if agent.__class__.__name__ == "ThompsonBernoulli" and t % 100 == 0:
            agent.plot_posteriors(title=f"Turn {t}", arms=4)

for agent in agent_list:
    plt.plot(agent.regret, label=agent.__class__.__name__)

plt.legend()
plt.show()

# %%
