# %%
import agents
from arms.bernoulli import BernoulliArm
from bandits import Bandits
from plot import plot_regret

# Create bandits
p_list = [0.3, 0.5, 0.4]

bandits = Bandits([BernoulliArm(p) for p in p_list])

# %%
agent_list: list[agents.Agent] = [
    agents.Random(bandits),
    agents.Greedy(bandits, test_rounds=1),
    agents.EpsilonGreedy(bandits, test_rounds=1, eps=0.1),
    agents.EpsilonDecay(bandits, test_rounds=1, eps_0=1),
    agents.UCB(bandits, test_rounds=10),
    agents.ThompsonBernoulli(bandits),
]

T = 1000

for agent in agent_list:
    agent.play(
        T,
        plot_at=[0, 10, 100, 999],
    )

plot_regret(agent_list)
plot_regret(agent_list, yscale="log")
print("\n" * 10)

# %%
