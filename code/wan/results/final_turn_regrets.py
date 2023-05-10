# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

PATH = Path("/Users/boyesjo/NTH/tma4900/code/wan/results/random3")


# %%
def get_last_turns(path):
    df = pd.read_parquet(path)
    max_turn = df.turn.max()
    df = df[df.turn == max_turn]
    # return last column as numpy array
    return df.iloc[:, -1].to_numpy()


df = pd.DataFrame(
    {
        "qucb": get_last_turns(PATH / "qucb.parquet"),
        "thomp": get_last_turns(PATH / "thomp.parquet"),
        "ucb": get_last_turns(PATH / "ucb.parquet"),
    }
)


# %%
# plot stacked histogram
def transform(x):
    return x
    # return np.log(x)


# plot stacked histogram, transform data, alpha=0.5
sns.histplot(
    data=df,
    bins=20,
    # stat="density",
    # common_norm=False,
    # alpha=0.2,
    binrange=(0, 600),
    element="step",
)
plt.show()

# %%
df.to_csv(PATH / "final_turn_regerts.dat", index=False)

# %%
# z test for difference in means
from scipy.stats import ttest_ind

for col1 in df.columns:
    for col2 in df.columns:
        if col1 == col2:
            continue
        print(f"{col1} vs {col2}")
        print(ttest_ind(df[col1], df[col2], equal_var=False))
        print()
# %%
