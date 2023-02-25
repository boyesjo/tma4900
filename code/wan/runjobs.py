import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from qucb1 import QUCB1
from scipy.stats import beta


def run_qucb1(
    filename: str,
    p_list: np.ndarray,
    horizon: int,
    delta: float,
) -> None:
    logger.info(f"{filename} started")
    qucb1 = QUCB1(p_list, C1=2)
    arms_played, times_played = qucb1.run(horizon, delta)
    logger.success(f"{filename} finished")
    pd.DataFrame(
        {
            "arm": np.repeat(arms_played, times_played),
            "regret": qucb1.regret(),
        }
    ).to_csv(Path("results") / "big2" / f"{filename}.csv", index_label="turn")


def run_thompson(filename: str, p_list: np.ndarray, horizon: int) -> None:
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
        Path("results") / "big2" / f"{filename}.csv", index_label="turn"
    )
    logger.success(f"{filename} finished")


if __name__ == "__main__":
    p_list = np.array([0.5, 0.505])
    horizon = 250_000
    # delta = 0.01

    tasks = np.arange(100)
    with multiprocessing.Pool() as pool:
        # pool.starmap(
        #     run_qucb1,
        #     [(f"qucb1_{task}", p_list, horizon, delta) for task in tasks],
        # )
        pool.starmap(
            run_thompson,
            [(f"thompson_{task}", p_list, horizon) for task in tasks],
        )
