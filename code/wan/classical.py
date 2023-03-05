from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import beta


def run_thompson(
    filename: str,
    folder: str,
    p_list: np.ndarray,
    horizon: int,
) -> None:
    logger.info(f"{filename} started")
    posteriors = [{"a": 1, "b": 1} for _ in p_list]
    regret = np.zeros(horizon)

    p_max = np.max(p_list)

    for t in range(horizon):
        arm = np.argmax([beta.rvs(**post) for post in posteriors])
        reward = np.random.binomial(1, p_list[arm])
        posteriors[arm]["a"] += reward
        posteriors[arm]["b"] += 1 - reward
        regret[t] = p_max - p_list[arm]

    regret = np.cumsum(regret)
    pd.DataFrame({"regret": regret}).to_csv(
        Path("results") / folder / f"{filename}.csv",
        index=False,
    )
    logger.success(f"{filename} finished")


def run_ucb(
    filename: str,
    folder: str,
    p_list: np.ndarray,
    horizon: int,
) -> None:

    logger.info(f"{filename} started")

    n_arms = len(p_list)
    est_list = np.zeros_like(p_list)
    times_pulled = np.zeros_like(p_list)
    reg = []

    for arm in range(n_arms):
        est_list[arm] = np.random.binomial(1, p_list[arm])
        times_pulled[arm] += 1
        reg.append(max(p_list) - p_list[arm])

    for t in range(n_arms, horizon):
        arm = np.argmax(est_list + np.sqrt(2 * np.log(t) / times_pulled))
        reg.append(max(p_list) - p_list[arm])

        reward = np.random.binomial(1, p_list[arm])
        est_list[arm] = (est_list[arm] * times_pulled[arm] + reward) / (
            times_pulled[arm] + 1
        )

        times_pulled[arm] += 1

    regret = np.cumsum(np.asarray(reg))

    pd.DataFrame({"regret": regret}).to_csv(
        Path("results") / folder / f"{filename}.csv",
        index=False,
    )
    logger.success(f"{filename} finished")
