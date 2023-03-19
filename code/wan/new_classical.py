from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import beta


def run_thompson(
    filename: str,
    p_list: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
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
    logger.success(f"{filename} finished")
    return pd.DataFrame({"regret": regret})


def run_ucb(
    filename: str,
    p_list: np.ndarray,
    horizon: int,
) -> pd.DataFrame:

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

    df = pd.DataFrame({"regret": regret})
    logger.success(f"{filename} finished")
    return df


def lai_robbins_bound(p_list: np.ndarray, horizon: int) -> pd.DataFrame:
    p_max = np.max(p_list)
    subopt = p_max - p_list
    kl = p_list * np.log(p_list / p_max) + (1 - p_list) * np.log(
        (1 - p_list) / (1 - p_max)
    )
    coeff = sum(subopt / kl for subopt, kl in zip(subopt, kl) if kl != 0)
    regret = coeff * np.log(np.arange(1, horizon + 1))
    df = pd.DataFrame({"regret": regret})
    df.index.name = "turn"
    return df


def thompson_bound(p_list: np.ndarray, horizon: int) -> pd.DataFrame:
    pass


def ucb_bound(p_list: np.ndarray, horizon: int) -> pd.DataFrame:
    p_max = np.max(p_list)
    subopt = p_max - p_list
    coeff = 8 * sum(1 / subopt for subopt in subopt if subopt != 0)
    regret = coeff * np.log(np.arange(1, horizon + 1))
    df = pd.DataFrame({"regret": regret})
    df.index.name = "turn"
    return df
