import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from qucb1 import QUCB1


def run_qucb1(
    filename: str,
    p_list: np.ndarray,
    horizon: int,
    delta: float = 0.1,
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
    ).to_csv(Path("results") / f"{filename}.csv", index_label="turn")


if __name__ == "__main__":
    p_list = np.array([0.5, 0.501])
    horizon = int(1e5)
    delta = 0.1

    tasks = np.arange(100)
    with multiprocessing.Pool() as pool:
        pool.starmap(
            run_qucb1,
            [(f"qucb1_{task}", p_list, horizon, delta) for task in tasks],
        )
