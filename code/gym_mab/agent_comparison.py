# %%
import agents
import numpy as np
from bandits import BanditsEnv
from stable_baselines3 import DQN

n_arms = 3
max_turns = 1000

env = BanditsEnv(n_arms, max_turns)

# %%
agent_dict = {
    "random": agents.Random(env),
    "greedy": agents.Greedy(env),
    "epsilon_greedy": agents.EpsilonGreedy(env, eps=0.1),
    "epsilon_decay": agents.EpsilonDecay(env, eps_0=1),
    "ucb": agents.UCB(env),
    "thompson": agents.ThompsonBernoulli(env),
}

regrets = {name: np.zeros(env.max_turns) for name in agent_dict.keys()}

# %%
for name, agent in agent_dict.items():

    obs = env.reset(keep_p_list=True)

    done = False
    while not done:
        action, _ = agent.predict(obs)
        obs, reward, done, _ = env.step(action)
    regrets[name] = env.regret

# %%
import matplotlib.pyplot as plt

print(env.p_list)

for name, regret in regrets.items():
    plt.plot(regret, label=name)
plt.legend()
plt.yscale("log")
plt.show()

# %%
