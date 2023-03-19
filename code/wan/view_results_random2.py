# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


# %%
FOLDER = Path("results") / "random2"


def load_sims() -> pd.DataFrame:
    df_list = []
    for sim in range(1000):
        try:
            df = pd.read_parquet(str(FOLDER / f"sim_{sim}.parquet"))
            df["turn"] = df.index
            df["sim"] = sim
            df_list.append(df)
        except FileNotFoundError:
            pass
    df = pd.concat(df_list)
    return df


df = load_sims()

# %%
# save as seperate parquet files
df[["turn", "sim", "thompson"]].to_parquet(
    FOLDER / "thomp.parquet", index=False, compression="brotli"
)

# %%
# pivot longer
long = df.melt(
    id_vars=["sim", "turn"], var_name="algorithm", value_name="regret"
).set_index(["sim", "turn", "algorithm"])

# %%
RESOLUTION = 100
sns.lineplot(
    data=long.loc[
        long.index.get_level_values("turn").isin(
            np.linspace(0.1, 250_000, RESOLUTION, dtype=int)
        )
    ],
    x="turn",
    y="regret",
    hue="algorithm",
    errorbar="ci",
    # errorbar="sd",
    # errorbar=lambda x: (np.quantile(x, 0.05), np.quantile(x, 0.95)),
)

# %%
# %%
# same but loglog
# only turns in np.geomspace(0.1, 250_000, 100).astype(int)

RESOLUTION = 100
sns.lineplot(
    data=long.loc[
        long.index.get_level_values("turn").isin(
            np.geomspace(0.1, 250_000, RESOLUTION, dtype=int)
        )
    ],
    x="turn",
    y="regret",
    hue="algorithm",
    errorbar="ci",
    # errorbar="sd",
    # errorbar=lambda x: (np.quantile(x, 0.05), np.quantile(x, 0.95)),
)
plt.xscale("log")
plt.yscale("log")
# %%
