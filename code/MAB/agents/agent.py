from abc import ABC, abstractmethod

import numpy as np
from bandits import Bandits


class Agent(ABC):
    def __init__(self, bandits: Bandits, *, test_rounds: int = 0):
        self.rewards: list[float] = []
        self.arms_pulled: list[int] = []
        self.arm_counts: list[int] = [0] * bandits.n_arms
        self.arm_rewards: list[list[float]] = [
            [] for _ in range(bandits.n_arms)
        ]
        self.turns = 0
        self.bandits = bandits
        self.test_rounds = test_rounds

    @abstractmethod
    def _select_arm(self) -> int:
        pass

    def pull(self) -> float:

        # run test rounds, try each arm
        if self.turns < self.test_rounds * self.bandits.n_arms:
            arm_idx = self.turns % self.bandits.n_arms
        else:
            arm_idx = self._select_arm()

        reward = self.bandits.pull(arm_idx)
        self.rewards.append(reward)
        self.arms_pulled.append(arm_idx)

        self.arm_counts[arm_idx] += 1
        self.arm_rewards[arm_idx].append(reward)

        self.turns += 1

        return reward

    def play(self, T: int, **kwargs) -> None:
        for _ in range(T):
            self.pull()

    @property
    def arm_means(self) -> list[float]:
        return [np.mean(rewards) for rewards in self.arm_rewards]

    def reset(self) -> None:
        self.rewards = []
        self.arms_pulled = []

    @property
    def regret(self) -> float:
        means_of_armes_pulled = np.array(
            [self.bandits.arms[arm_idx].mean for arm_idx in self.arms_pulled]
        )
        return np.cumsum(self.bandits.best_arm_mean - means_of_armes_pulled)
