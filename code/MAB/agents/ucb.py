import numpy as np
from bandits import Bandits

from .agent import Agent


class UCB(Agent):
    def __init__(
        self,
        bandits: Bandits,
        *,
        alpha: float = 1,
        test_rounds: int = 1,
    ):
        super().__init__(bandits, test_rounds=test_rounds)
        self.alpha = alpha

    def _select_arm(self) -> int:
        coeff = np.sqrt(
            2
            * self.alpha
            * np.log(self.turns + 1)
            / (1 + np.array(self.arm_counts))
        )
        return np.argmax(self.arm_means + coeff)  # type: ignore

        # return np.argmax(
        #     self.arm_means
        #     * np.sqrt(
        #         2
        #         * self.alpha
        #         * np.log(self.turns + 1)
        #         / (1 + np.array(self.arm_counts))
        #     )
        # )
