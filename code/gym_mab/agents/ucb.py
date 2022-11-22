import numpy as np
from bandits import BanditsEnv

from .agent import Agent


class UCB(Agent):
    def __init__(
        self,
        env: BanditsEnv,
        *,
        alpha: float = 1,
        test_rounds: int = 1,
    ):
        super().__init__(env, test_rounds=test_rounds)
        self.alpha = alpha

    def _select_arm(self) -> int:

        return np.argmax(
            self.arm_means
            * np.sqrt(
                2
                * self.alpha
                * np.log(self.turns + 1)
                / (1 + np.array(self.arm_counts))
            )
        )
