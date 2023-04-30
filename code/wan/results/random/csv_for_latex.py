# %%
import pandas as pd
from pathlib import Path
import numpy as np


FOLDER = Path()

# %%
df_qucb = pd.read_parquet(FOLDER / "qucb.parquet").set_index("turn")
df_ucb = pd.read_parquet(FOLDER / "ucb.parquet").set_index("turn")
df_thomp = (
    pd.read_parquet(FOLDER / "thomp.parquet")
    .set_index("turn")
    .rename({"thompson": "thomp"}, axis=1)
)

HORIZON = df_qucb.index.max() + 1

# %%
# print number of simulations for each df
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    n_sims = df["sim"].nunique()
    print(f"{name}: {n_sims}")

# %%
df_all = pd.DataFrame()
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    # grab 200 turns evenly spaced
    df = df.iloc[:: int(HORIZON / 200)]
    df_all[f"{name}_mean"] = df.groupby("turn")[name].mean()

df_all.to_csv(FOLDER / "plot_data.dat")

# %%
# same but turns spaced on log scale
turns = np.geomspace(1, HORIZON - 1, 200).astype(int)

# if duplicate turns, set to higest not in list
for i in range(1, len(turns)):
    if turns[i] in turns[:i]:
        turns[i] = turns[i - 1] + 1
turns

df_all = pd.DataFrame()
for df, name in [
    (df_qucb, "qucb"),
    (df_ucb, "ucb"),
    (df_thomp, "thomp"),
]:
    df = df.loc[turns]
    df_all[f"{name}_mean"] = df.groupby("turn")[name].mean()

df_all.to_csv(FOLDER / "plot_data_log.dat")

# %%
df_all.plot(loglog=True)


# %%
def curve_fit():
    # fit curves to extrapolate
    from scipy.optimize import curve_fit

    def f_sqrt(x, a, b):
        return a * np.sqrt(x) + b

    def f_log(x, a, b):
        return a * np.log(x) + b

    new_idx = np.geomspace(1, 1e10, 1000).astype(int)
    df_extrap = pd.DataFrame(index=new_idx)

    for name in ["qucb", "ucb", "thomp"]:
        df = df_all[f"{name}_mean"]
        # fit curve on loglog scale
        popt, pcov = curve_fit(f_log, np.log(df.index), np.log(df))
        # plot fitted curve
        df_extrap[f"{name}_sqrt"] = np.exp(
            f_sqrt(np.log(df_extrap.index), *popt)
        )

    # plot extrapolated curves on top of data
    df_extrap.plot(loglog=True)
    df_all.plot(loglog=True)


# %%
