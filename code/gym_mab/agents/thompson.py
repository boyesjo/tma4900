from abc import abstractmethod
from typing import Optional, Sequence

import numpy as np
from bandits import BanditsEnv
from scipy.stats import beta, rv_continuous, rv_discrete

from .agent import Agent


class Thompson(Agent):
    @abstractmethod
    def posterior(self) -> list[rv_continuous | rv_discrete]:
        pass

    def _select_arm(self) -> int:
        samples = [self.posterior(i).rvs() for i in range(self.n_arms)]
        return np.argmax(samples)


class ThompsonBernoulli(Thompson):
    def posterior(self, arm: int) -> rv_discrete:
        times_pulled = self.arm_counts[arm]
        successes = self.env.arm_rewards[arm]
        return beta(
            a=1 + successes,
            b=1 + times_pulled - successes,
        )
