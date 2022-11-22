from abc import ABC, abstractmethod

import numpy as np
from bandits import BanditsEnv


class Agent(ABC):
    def __init__(self, env: BanditsEnv, *, test_rounds: int = 0):
        self.env = env
        self.n_arms = env.action_space.n
        self.test_rounds = test_rounds

    @abstractmethod
    def _select_arm(self) -> int:
        pass

    @property
    def turns(self) -> int:
        return self.env.turn

    @property
    def rewards(self) -> np.ndarray:
        return np.array(self.env.reward_list)

    @property
    def arm_means(self) -> np.ndarray:
        return self.env.arm_means

    @property
    def arm_counts(self) -> np.ndarray:
        return np.array(self.env.arm_counts)

    @property
    def arms_pulled(self) -> np.ndarray:
        return np.array(self.env.arms_pulled)

    def predict(self, obs, **kwargs) -> tuple[int, None]:

        if self.turns < self.test_rounds * self.n_arms:
            return self.turns % self.n_arms, None

        return self._select_arm(), None
