# %%
from typing import Optional

import gym
import numpy as np
from gym import spaces


class BanditsEnv(gym.Env):

    metadata = {"render.modes": ["print", "pyplot"]}

    def __init__(self, arms: int, max_turns: int = 100):
        self.amrs = arms
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                2,
                arms,
            ),
            dtype=np.float32,
        )

    def reset(self, seed=None, **kwargs) -> np.ndarray:
        self.arms_pulled: list[int] = []
        self.arm_counts = np.zeros(self.amrs)
        self.arm_rewards = np.zeros(self.amrs)
        if not kwargs.get("keep_p_list", False) or not hasattr(self, "p_list"):
            self.p_list = np.random.uniform(0, 1, self.amrs)
        self.turn = 0
        self.reward_list: list[float] = []
        return self._get_obs(0)

    def _get_obs(self, action: int) -> np.ndarray:
        return np.array([self.arm_means, self.arm_counts])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if action == 0:  # explore
            action = np.random.randint(self.amrs)
        else:  # exploit
            action = np.argmax(self.arm_means)
        reward = np.random.binomial(1, self.p_list[action])

        self.turn += 1
        self.arms_pulled.append(action)
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward

        self.reward_list.append(reward)
        # reward = sum(self.reward_list)

        done = self.turn >= self.max_turns

        obs = self._get_obs(action)
        return obs, reward, done, {}

    def render(
        self, mode: str = "print", action: Optional[int] = None, **kwargs
    ) -> None:
        if mode == "print":
            print(
                # self.p_list,
                action if action is not None else "",
                np.argmax(self.p_list),
                # self.reward_list[-1],
                sep="\t",
            )

        if mode == "pyplot":
            import matplotlib.pyplot as plt

            regret = self.regret
            # plt.title(f"Arm means: {self.p_list}")
            plt.plot(regret)
            plt.show()

    @property
    def arm_means(self) -> np.ndarray:
        means = self.arm_rewards / self.arm_counts
        means[np.isnan(means)] = 0
        return means

    @property
    def regret(self) -> float:
        return np.cumsum(np.max(self.p_list) - self.p_list[self.arms_pulled])


# %%
import stable_baselines3 as sb3

N = 20
T = 100

agents = {
    "A2C": sb3.A2C,
    "PPO": sb3.PPO,
    "DQN": sb3.DQN,
}

env = BanditsEnv(arms=3, max_turns=T)

regrets = {name: np.zeros((N, T)) for name in agents.keys()}

for name, agent in agents.items():

    model = agent(
        "MlpPolicy",
        env,
        # verbose=1,
        # gamma=0,
    )
    model.learn(total_timesteps=500_000)

    obs = env.reset()
    for i in range(N * T):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render(action=action)
        if done:
            j = i // T
            regrets[name][j] = env.regret
            obs = env.reset()
#%%

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
plt.savefig("regrets_ml.png")
plt.show()

# %%
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
).to_csv("regrets_ml.csv", index_label=["n", "t"])

# %%
