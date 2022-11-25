from typing import Optional

import gym
import numpy as np
from gym import spaces


class BanditsEnv(gym.Env):

    metadata = {"render.modes": ["print", "pyplot"]}

    def __init__(self, arms: int, max_turns: int = 100):
        self.amrs = arms
        self.max_turns = max_turns

        self.action_space = spaces.Discrete(arms)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                3,
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
        arm_means = self.arm_means
        reward_array = np.zeros(self.amrs)
        reward_array[action] = self.reward_list[-1] if self.reward_list else 0

        return np.array([arm_means, self.arm_counts, reward_array])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        reward = np.random.binomial(1, self.p_list[action])

        self.turn += 1
        self.arms_pulled.append(action)
        self.arm_counts[action] += 1
        self.arm_rewards[action] += reward

        self.reward_list.append(reward)
        reward = sum(self.reward_list)

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

    @property
    def arm_means(self) -> np.ndarray:
        return self.arm_rewards / self.arm_counts

    @property
    def regret(self) -> np.ndarray:
        return np.cumsum(np.max(self.p_list) - self.p_list[self.arms_pulled])
