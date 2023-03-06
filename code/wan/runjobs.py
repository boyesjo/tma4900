import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from classical import run_thompson, run_ucb
from loguru import logger
from qucb1 import QUCB1

FOLDER = "easy"
P_LIST = np.array([0.5, 0.51])
HORIZON = 100_000
DELTA = 0.01
N_SIMULATIONS = 100


def run_qucb1(
    filename: str,
    folder: str,
    p_list: np.ndarray,
    horizon: int,
    delta: float,
) -> None:
    logger.info(f"{filename} started")
    qucb1 = QUCB1(p_list, C1=2)
    arms_played, times_played = qucb1.run(horizon, delta)
    logger.success(f"{filename} finished")
    pd.DataFrame({"regret": np.cumsum(qucb1.regret()[:horizon])}).to_csv(
        Path("results") / folder / f"{filename}.csv",
        index=False,
    )


def run_all(
    sim: int,
    folder: str,
    p_list: np.ndarray,
    horizon: int,
    delta: float = 0.01,
) -> None:
    # kys unix
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    logger.debug(f"{sim=}, seed: {np.random.get_state()[1][0]}")

    # run_qucb1(f"qucb_{sim}", folder, p_list, horizon, delta)
    run_thompson(f"thompson_{sim}", folder, p_list, horizon)
    run_ucb(f"ucb_{sim}", folder, p_list, horizon)


if __name__ == "__main__":

    (Path("results") / FOLDER).mkdir(parents=True, exist_ok=True)

    tasks = np.arange(N_SIMULATIONS)
    with multiprocessing.Pool() as pool:
        pool.starmap(
            run_all,
            [
                (
                    sim,
                    FOLDER,
                    P_LIST,
                    HORIZON,
                    DELTA,
                )
                for sim in tasks
            ],
        )
