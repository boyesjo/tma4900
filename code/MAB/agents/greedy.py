from typing import Callable

import numpy as np
from bandits import Bandits

from .agent import Agent


class Greedy(Agent):
    def __init__(self, bandits: Bandits, *, test_rounds: int = 1):
        super().__init__(bandits, test_rounds=test_rounds)

    def _select_arm(self) -> int:

        return np.argmax(self.arm_means)


class EpsilonGreedy(Greedy):
    def __init__(
        self,
        bandits: Bandits,
        *,
        eps: float = 0.1,
        test_rounds: int = 1,
        **kwargs
    ):
        if not 0 <= eps <= 1:
            raise ValueError("Epsilon must be between 0 and 1")

        self.eps = eps
        super().__init__(bandits, test_rounds=test_rounds, **kwargs)

    def _select_arm(self) -> int:

        if np.random.random() < self.eps:
            return np.random.randint(self.bandits.n_arms)

        return super()._select_arm()


DecayFunction = Callable[[float, int], float]


class EpsilonDecay(EpsilonGreedy):
    def __init__(
        self,
        bandits: Bandits,
        *,
        eps_0: float = 0.1,
        test_rounds: int = 1,
        decay: DecayFunction = lambda eps_0, t: eps_0 / (t + 1),
        **kwargs
    ):
        super().__init__(bandits, test_rounds=test_rounds, **kwargs)
        if eps_0 < 0:
            raise ValueError("Epsilon_0 must be positive")

        self.eps_0 = eps_0
        self.eps = eps_0
        self.decay = decay

    def _select_arm(self) -> int:
        self.eps = self.decay(self.eps_0, self.turns)
        return super()._select_arm()
