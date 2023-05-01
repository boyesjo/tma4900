import multiprocessing
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from new_classical import run_thompson, run_ucb
from loguru import logger
from qucb1 import QUCB1
from scipy.stats import beta, norm

settings = {
    "folder": "random2",
    "horizon": 250_000,
    "delta": 0.01,
    "n_simulations": 1000,
}


def run_qucb1(
    name: str,
    p_list: np.ndarray,
    horizon: int,
    delta: float,
) -> pd.DataFrame:
    logger.info(f"{name} started")
    qucb1 = QUCB1(p_list, C1=2)
    arms_played, times_played = qucb1.run(horizon, delta)
    logger.success(f"{name} finished")
    df = pd.DataFrame({"regret": np.cumsum(qucb1.regret()[:horizon])})
    return df


def gen_p_list():
    a = 2
    b = 2
    sigma = 0.02
    p1 = beta.rvs(a, b)
    logit_p1 = np.log(p1 / (1 - p1))
    logit_p2 = logit_p1 + norm.rvs(scale=sigma)
    p2 = 1 / (1 + np.exp(-logit_p2))
    return np.array([p1, p2])


def run_all(
    p_list: np.ndarray,
    sim: int,
    folder: Path,
    horizon: int,
    delta: float = 0.01,
) -> None:
    # kys unix
    np.random.seed((os.getpid() * int(time.time())) % 123456789)
    logger.info(f"{sim=}, {p_list=}")

    df_qucb = run_qucb1(f"qucb_{sim}", p_list, horizon, delta)
    df_thompson = run_thompson(f"thompson_{sim}", p_list, horizon)
    df_ucb = run_ucb(f"ucb_{sim}", p_list, horizon)

    # join and rename columns
    df = pd.concat([df_qucb, df_thompson, df_ucb], axis=1)
    df.columns = ["qucb", "thompson", "ucb"]
    df.to_parquet(
        folder / f"sim_{sim}.parquet", index=False, compression="brotli"
    )
    logger.success(f"{sim=} finished and saved to disk. {p_list=}")


if __name__ == "__main__":
    folder = Path("results") / settings["folder"]
    folder.mkdir(parents=True, exist_ok=True)

    # save settings
    with open(folder / "settings.pickle", "wb") as f:
        pickle.dump(settings, f)
    with open(folder / "settings.txt", "w") as f:
        f.write(str(settings))

    # generate p_lists
    df_p_list = pd.DataFrame(
        [gen_p_list() for _ in range(settings["n_simulations"])],
        columns=["p1", "p2"],
    )
    df_p_list.to_parquet(
        folder / "p_list.parquet", index=False, compression="brotli"
    )

    # tasks = np.arange(settings["n_simulations"])
    tasks = np.array(
        [
            206,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            290,
            291,
            292,
            293,
            294,
            295,
            296,
            398,
            399,
            400,
            401,
            402,
            403,
            404,
            547,
            548,
            885,
            886,
            887,
            888,
            889,
            890,
            979,
            980,
        ]
    )
    with multiprocessing.Pool() as pool:
        pool.starmap(
            run_all,
            [
                (
                    df_p_list.iloc[sim].values,
                    sim,
                    folder,
                    settings["horizon"],
                    settings["delta"],
                )
                for sim in tasks
            ],
        )
