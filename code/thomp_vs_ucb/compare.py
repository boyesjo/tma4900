# %%
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from thompson import thompson
from ucb import ucb

N_SIMS = 100
HORIZON = 250_000
P1 = np.array(
    [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        0.95,
        0.99,
        0.995,
        0.999,
    ]
)
DELTA = np.array(
    [
        0.2,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001,
        0.0005,
        0.0001,
    ]
)
P_LIST_LIST = np.array([(p1, p1 - delta) for delta in DELTA for p1 in P1])


# %%
def compare(sim: int) -> None:
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    logger.info(f"Starting sim {sim}")
    ucb_regret = [ucb(p_list, HORIZON)[-1] for p_list in P_LIST_LIST]
    thomp_regret = [thompson(p_list, HORIZON)[-1] for p_list in P_LIST_LIST]
    df = pd.DataFrame(
        {
            "p1": P_LIST_LIST[:, 0],
            "p2": P_LIST_LIST[:, 1],
            "ucb_regret": ucb_regret,
            "thomp_regret": thomp_regret,
        }
    )
    df.to_csv(Path("results") / f"results_{sim}.csv", index=False)
    logger.success(f"Finished sim {sim}")


def runjobs() -> None:
    logger.info("Starting")

    jobs = np.arange(N_SIMS)
    with mp.Pool(processes=mp.cpu_count() - 1) as pool:
        pool.map(compare, jobs)
    logger.success("Finished")


# %%
if __name__ == "__main__":
    runjobs()
