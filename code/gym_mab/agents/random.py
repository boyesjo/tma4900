import numpy as np

from .agent import Agent


class Random(Agent):
    def _select_arm(self) -> int:
        return np.random.randint(self.n_arms)
