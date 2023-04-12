# %%
from env import BernoulliBanditsEnv
import stable_baselines3 as sb3
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt


env = BernoulliBanditsEnv(
    min_turns=10,
    max_turns=1000,
    arms=2,
    prior=lambda arms: np.linspace(0.5, 0.505, arms),
)

agents = {
    # "A2C": sb3.A2C,
    # "PPO": sb3.PPO,
    "DQN": sb3.DQN,
}

total_timesteps = 250_000


class RewardCounter(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.regret = np.zeros(total_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps < total_timesteps + 1:
            # get regret of action
            self.regret[self.num_timesteps - 1] = self.locals["infos"][-1][
                "regret_action"
            ]
        return True


# %%
reward_counter = RewardCounter()
for name, agent in agents.items():
    obs = env.reset()
    model = agent("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=reward_counter)

    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, result, done, info = env.step(action)
        print(info["turn"], info["p_list"], action, info["times_pulled"])

# %%
plt.plot(np.cumsum(reward_counter.regret))

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
