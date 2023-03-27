# %%
from env import BernoulliBanditsEnv
import stable_baselines3 as sb3
import numpy as np


TURNS = 1000

env = BernoulliBanditsEnv(min_turns=TURNS, arms=2)

agents = {
    # "A2C": sb3.A2C,
    # "PPO": sb3.PPO,
    "DQN": sb3.DQN,
}

# %%
for name, agent in agents.items():
    obs = env.reset()
    model = agent("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500_000)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, result, done, info = env.step(action)
        print(info["turn"], info["p_list"], action, info["times_pulled"])
# %%
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, result, done, info = env.step(action)
    print(info["turn"], info["p_list"], action, info["times_pulled"])
    done = info["turn"] > 10_000

# %%
# play 100 times, save and plot regret
import matplotlib.pyplot as plt

turns = 20_000
N = 10

regrets = np.zeros((N, turns + 1))

for i in range(N):
    regret = np.zeros(turns + 1)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, result, done, info = env.step(action)
        regret[info["turn"] - 1] = info["regret"]
        done = info["turn"] > turns
    regrets[i] = regret

# %%
plt.plot(regrets.mean(axis=0))
# %%
for reg in regrets:
    plt.plot(reg)
    plt.show()

# %%
