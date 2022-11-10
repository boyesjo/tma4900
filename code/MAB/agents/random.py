import numpy as np
from bandits import Bandits

from .agent import Agent


class Random(Agent):
    def _select_arm(self) -> int:

        return np.random.randint(self.bandits.n_arms)
