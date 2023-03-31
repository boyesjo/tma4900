# %%
from env import BernoulliBanditsEnv
import stable_baselines3 as sb3
import numpy as np


env = BernoulliBanditsEnv(
    min_turns=10,
    max_turns=1000,
    arms=2,
    prior=lambda arms: np.linspace(0.1, 0.9, arms),
)

agents = {
    # "A2C": sb3.A2C,
    "PPO": sb3.PPO,
    # "DQN": sb3.DQN,
}

# %%
for name, agent in agents.items():
    obs = env.reset()
    model = agent("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100_000)

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

turns = 1_000
N = 10

regrets = np.zeros((N, turns + 1))

for i in range(N):
    regret = np.zeros(turns + 1)
    obs = env.reset()
    env.p_list = np.array([0.9, 0.1])
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
