import numpy as np
from arms.arm import Arm


class Bandits:
    def __init__(self, arms: list[Arm]):
        self.n_arms = len(arms)
        self.arms = arms

    def __str__(self):
        return f"Bandits({self.arms})"

    def __getitem__(self, item: int) -> Arm:
        return self.arms[item]

    def pull(self, arm_idx: int) -> float:
        return self.arms[arm_idx].pull()

    @property
    def best_arm(self) -> int:
        return np.argmax([arm.mean for arm in self.arms])

    @property
    def best_arm_mean(self) -> float:
        return self.arms[self.best_arm].mean
