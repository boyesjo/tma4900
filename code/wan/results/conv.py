import glob
from pathlib import Path

import pandas as pd
from loguru import logger

folder = "low_prob_fix2"
prefixes = [
    "thompson",
    "ucb",
    "qucb",
]

for prefix in prefixes:
    df_list = []
    paths = glob.glob(str(Path() / folder / f"{prefix}_*.csv"))
    for path in paths:
        logger.info(path)
        df = pd.read_csv(path)
        df["turn"] = df.index
        df["sim"] = path.split("_")[-1].split(".")[0]
        df.set_index(["turn", "sim"], inplace=True)
        df_list.append(df)

    df = pd.concat(df_list)
    df.reset_index().to_parquet(
        Path() / folder / f"{prefix}.parquet",
        index=False,
        compression="brotli",
    )
    logger.success(f"{prefix} done")
