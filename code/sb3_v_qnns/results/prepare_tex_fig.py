import pandas as pd
import numpy as np

df = pd.read_parquet("rl.parquet").reset_index()


turns = np.linspace(0, 250_000 - 1, 200, dtype=int)
df = df[df.turn.isin(turns)]

df.groupby("turn").mean().drop(columns=["sim"]).to_csv("rl.dat")
