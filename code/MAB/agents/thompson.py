from abc import abstractmethod
from typing import Optional, Sequence

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
        super().__init__(bandits)

    @abstractmethod
    def _update_posterior_params(self, arm_idx: int, reward: float) -> None:
        pass

    def plot_posteriors(
        self,
        *,
        arms: Optional[list[int] | int] = None,
        num_samples: int = 1000,
        title: str = "",
        **kwargs,
    ) -> None:

        import matplotlib.pyplot as plt

        if isinstance(arms, int):
            num_arms = min(self.bandits.n_arms, arms)
            arms = list(range(self.bandits.n_arms))[:num_arms]

        elif arms is None:
            num_arms = min(self.bandits.n_arms, 5)
            arms = list(range(self.bandits.n_arms))[:num_arms]

        fig, axs = plt.subplots(1, len(arms), figsize=(10, 5))
        for i, ax in enumerate(axs):

            ax.axvline(
                self.bandits.arms[arms[i]].mean,
                color="black",
                linestyle="--",
                label="true mean",
            )

            x = np.linspace(0, 1, num_samples)
            ax.plot(x, self.posterior_func.pdf(x, **self.posterior_params[i]))
            ax.set_title(f"Arm {i}, pulled {self.arm_counts[i]} times")

        fig.suptitle(title)
        plt.show()

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

    def play(self, T: int, plot_at: list[int] = [], **kwargs) -> None:

        # ensure plot_at values in [0, T]
        if any(i < 0 or i > T for i in plot_at):
            raise ValueError(
                f"plot_at values must be in [0, T], not {plot_at}"
            )

        for i in range(T):
            if i in plot_at:
                self.plot_posteriors(title=f"Turn {i}", **kwargs)
            self.pull()


class ThompsonBernoulli(Thompson):
    def __init__(self, bandits: Bandits):
        posterior_func = beta
        # initalise with flat beta(1,1) prior
        posterior_params = [{"a": 1, "b": 1} for _ in range(bandits.n_arms)]
        super().__init__(bandits, posterior_func, posterior_params)

    def _update_posterior_params(self, arm_idx: int, reward: float) -> None:
        self.posterior_params[arm_idx]["a"] += reward
        self.posterior_params[arm_idx]["b"] += 1 - reward
