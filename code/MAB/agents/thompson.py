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

        fig, axs = plt.subplots(1, len(arms), figsize=(20, 5))
        for i, ax in enumerate(axs):

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

    def play(self, T: int, **kwargs) -> None:

        plot_at = kwargs.pop("plot_at", None)

        if plot_at is not None:
            if isinstance(plot_at, int):
                plot_at = [plot_at]
            elif not isinstance(plot_at, Sequence):
                raise TypeError(
                    "plot_at must be an int or a sequence of ints, "
                    f"not {type(plot_at)}"
                )
            elif not all(isinstance(i, int) for i in plot_at):
                raise TypeError(
                    "plot_at must be an int or a sequence of ints, "
                    f"not {type(plot_at)}"
                )

            # ensure plot_at values in [0, T]
            if any(i < 0 or i > T for i in plot_at):
                raise ValueError(
                    f"plot_at values must be in [0, T], not {plot_at}"
                )

        else:
            plot_at = []

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
