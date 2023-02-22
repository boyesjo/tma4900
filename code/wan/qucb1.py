# %%
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from oracle import Oracle
from qmc import qmc

np.seterr(all="ignore")


# %%
class QUCB1:
    def __init__(
        self,
        p_list: np.ndarray,
        C1: float = 2,
    ):

        self.C1 = C1
        self.n_arms = len(p_list)
        self.p_list = p_list

        self.r_list = np.ones_like(p_list)
        self.oracle_list = [Oracle(p, 1) for p in p_list]
        self.estimate_list = np.zeros_like(p_list)

        self.end_explore_turn = None

        self.arms_played = []
        self.times_played = []

    def n_turns(self, r: float) -> int:
        return int(np.ceil(self.C1 * np.log(1 / self.delta) / r))

    def _log(self):
        logger.debug(f"arms played: {self.arms_played}")
        logger.debug(f"times played: {self.times_played}")
        logger.debug(f"estimates: {self.estimate_list}")

    def run(self, horizon: int, delta: float = 0.1):
        self.horizon = horizon
        self.delta = delta

        for arm in range(self.n_arms):
            oracle = self.oracle_list[arm]
            n = self.n_turns(self.r_list[arm])

            estimate, n_queries = qmc(oracle, n, delta, method="canonical")
            self.estimate_list[arm] = estimate
            self.arms_played.append(arm)
            self.times_played.append(n_queries)
            self._log()

        self.end_explore_turn = sum(self.times_played)
        logger.debug(f"end explore turn: {self.end_explore_turn}")

        while sum(self.times_played) < int(horizon):
            logger.debug(f"turn {sum(self.times_played)}, {self.horizon=}")
            arm = np.argmax(self.estimate_list + self.r_list)
            oracle = self.oracle_list[arm]
            self.r_list[arm] /= 2
            n = self.n_turns(self.r_list[arm])

            logger.debug(f"{arm=}, {n=}, {self.r_list=} {self.estimate_list=}")
            estimate, n_queries = qmc(oracle, n, delta, method="canonical")

            self.estimate_list[arm] = (estimate + self.estimate_list[arm]) / 2
            self.arms_played.append(arm)
            self.times_played.append(n_queries)
            self._log()

        return self.arms_played, self.times_played

    def regret(self) -> np.ndarray:
        max_p = max(self.p_list)
        regret_list = []
        for arm, n in zip(self.arms_played, self.times_played):
            p = self.p_list[arm]
            regret_list += [max_p - p] * n
        return np.asarray(regret_list)


# %%
def ucb(p_list: np.ndarray, horizon: float, delta: float = 0.1):
    # regular ucb1 alg for comparison
    n_arms = len(p_list)
    est_list = np.zeros_like(p_list)
    times_pulled = np.zeros_like(p_list)
    reg = []

    for arm in range(n_arms):
        est_list[arm] = np.random.binomial(1, p_list[arm])
        times_pulled[arm] += 1
        reg.append(max(p_list) - p_list[arm])

    for t in range(int(horizon) - n_arms):
        arm = np.argmax(est_list + np.sqrt(2 * np.log(t) / times_pulled))
        reg.append(max(p_list) - p_list[arm])

        reward = np.random.binomial(1, p_list[arm])
        est_list[arm] = (est_list[arm] * times_pulled[arm] + reward) / (
            times_pulled[arm] + 1
        )

        times_pulled[arm] += 1

    return np.asarray(reg)


# %%
if __name__ == "__main__":
    T = 1e5

    # p_list = np.random.uniform(0.01, 0.999, 16)
    p_list = np.array([0.5, 0.5 + 1e-2])
    # res = qucb1(p_list, T, delta=0.03)
    qucb = QUCB1(p_list)
    qucb.run(T, delta=0.03)

    # %%
    plt.plot(qucb.regret(), label="qucb1")
    plt.plot(ucb(p_list, T), label="ucb1")
    # plt.xscale("log")
    plt.legend()
    plt.savefig("2 arms p=0.5 and 0.51.png")
