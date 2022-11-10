from abc import abstractmethod

import numpy as np
from bandits import Bandits
from scipy.stats import beta, rv_continuous, rv_discrete

from .agent import Agent


class Thompson(Agent):
    def __init__(
        self,
        bandits: Bandits,
        posterior_func: rv_discrete | rv_continuous,
        posterior_params: list[dict[str, float]],
    ):
        self.posterior_params = posterior_params
        self.posterior_func = posterior_func
        self.rewards: list[float] = []
        self.arms_pulled: list[int] = []
        self.bandits = bandits
        super().__init__(bandits)

    @abstractmethod
    def _update_posterior_params(self, arm_idx: int, reward: float) -> None:
        pass

    def _select_arm(self) -> int:
        samples = [
            self.posterior_func.rvs(**params)
            for params in self.posterior_params
        ]

        return np.argmax(samples)

    def pull(self) -> float:
        reward = super().pull()
        self._update_posterior_params(self.arms_pulled[-1], reward)
        return reward


class ThompsonBernoulli(Thompson):
    def __init__(self, bandits: Bandits):
        posterior_func = beta
        posterior_params = [{"a": 1, "b": 1} for _ in range(bandits.n_arms)]
        super().__init__(bandits, posterior_func, posterior_params)

    def _update_posterior_params(self, arm_idx: int, reward: float) -> None:
        self.posterior_params[arm_idx]["a"] += reward
        self.posterior_params[arm_idx]["b"] += 1 - reward
