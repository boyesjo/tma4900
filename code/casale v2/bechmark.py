# %%
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qbai import QBAI
from qiskit import Aer, execute


# %%
def probs(qc: QBAI, shots: int = 10_000) -> dict[int, float]:
    alg = qc.copy()
    alg.add_measurements()
    counts = (
        execute(
            alg,
            backend=Aer.get_backend("aer_simulator"),
            shots=shots,
        )
        .result()
        .get_counts()
    )

    return {int(k, 2): v / shots for k, v in counts.items()}


def prob_correct(
    qc: QBAI,
    best_arm: int,
    shots: int = 10_000,
) -> float:
    counts = probs(qc, shots)
    return counts.get(best_arm, 0.0)


def probs_statevec(qc: QBAI) -> dict[int, float]:
    statevec = (
        execute(
            qc,
            backend=Aer.get_backend("statevector_simulator"),
        )
        .result()
        .get_statevector()
    )
    prob = np.abs(np.asarray(statevec)) ** 2
    new_prob = {x: 0.0 for x in range(2**qc.x_len)}
    for x in range(2**qc.x_len):
        for y in range(2**qc.y_len):
            new_prob[x] += prob[x << qc.y_len | y]
    return new_prob


def prob_correct_statevec(
    qc: QBAI,
    best_arm: int,
) -> float:
    counts = probs_statevec(qc)
    return counts.get(best_arm, 0.0)


# %%
p_list = np.linspace(0.0, 0.1, 32)
qc = QBAI(
    x_len=5,
    y_len=2,
    p_list=p_list,
)
counts = probs_statevec(qc)
plt.title(f"n = {qc.n}")
plt.bar(counts.keys(), counts.values())
plt.axhline(1 / np.size(p_list), color="green")
plt.axhline(np.max(p_list) / np.sum(p_list), color="red")
plt.show()


# %%
def test_ideal_n(
    p_max_list: Iterable[float],
    n_arms: int = 16,
) -> pd.DataFrame:

    res = []

    for p_max in p_max_list:
        res.append(
            {
                "p_max": p_max,
                "int": QBAI.ideal_n(np.linspace(0.0, p_max, n_arms)),
                "exact": QBAI.ideal_n_exact(np.linspace(0.0, p_max, n_arms)),
            }
        )

    return pd.DataFrame(res).set_index("p_max")


test_ideal_n(np.linspace(1e-5, 1, 100)).plot(loglog=True)


# %%
def test_n(
    n_list: Iterable[int],
    p_list: np.ndarray,
) -> pd.DataFrame:

    res = []
    x_len = np.log2(p_list.size).astype(int)
    assert 2**x_len == p_list.size
    for n in n_list:
        qc = QBAI(
            x_len=x_len,
            y_len=1,
            p_list=p_list,
            n=n,
        )
        res.append(
            {
                "n": n,
                "ideal_n": QBAI.ideal_n_exact(p_list),
                "simulated": prob_correct(qc, np.argmax(p_list)),
                "simulated_statevec": prob_correct_statevec(
                    qc, np.argmax(p_list)
                ),
                "random": 1 / np.size(p_list),
                "theoretical": np.max(p_list) / np.sum(p_list),
            }
        )

    return pd.DataFrame(res).set_index("n")


df = test_n(
    np.arange(1, 20),
    p_list=np.linspace(0.0, 1, 32),
)

df.drop("ideal_n", axis=1).plot()
plt.axvline(df["ideal_n"].iloc[0], color="black", linestyle="--")
plt.show()


# %%
def test_num_arms(
    n_arms_list: Iterable[int],
) -> pd.DataFrame:

    results = []

    for n_arms in n_arms_list:
        x_len = np.log2(n_arms).astype(int)
        assert 2**x_len == n_arms
        p_list = np.linspace(0.0, 0.05, n_arms)
        qc = QBAI(
            x_len=x_len,
            y_len=1,
            p_list=p_list,
        )
        results.append(
            {
                "n_arms": n_arms,
                "n": qc.n,
                "simulated": prob_correct(qc, np.argmax(p_list)),
                "simulated_statevec": prob_correct_statevec(
                    qc, np.argmax(p_list)
                ),
                "random": 1 / np.size(p_list),
                "theoretical": np.max(p_list) / np.sum(p_list),
            }
        )

    return pd.DataFrame(results).set_index("n_arms")


df = test_num_arms([2**i for i in range(1, 8)])
df.drop("n", axis=1).plot(loglog=True)


# %%
def test_arm_range(
    max_p_list: Iterable[float],
    n_arms: int = 16,
) -> pd.DataFrame:

    results = []
    x_len = np.log2(n_arms).astype(int)
    assert 2**x_len == n_arms

    for max_p in max_p_list:
        p_list = np.linspace(0.0, max_p, n_arms)
        qc = QBAI(
            x_len=x_len,
            y_len=3,
            p_list=p_list,
        )
        results.append(
            {
                "max_p": max_p,
                "n": qc.n,
                "simulated": prob_correct(qc, np.argmax(p_list)),
                "simulated_statevec": prob_correct_statevec(
                    qc, np.argmax(p_list)
                ),
                "random": 1 / np.size(p_list),
                "theoretical": np.max(p_list) / np.sum(p_list),
            }
        )

    return pd.DataFrame(results).set_index("max_p")


df = test_arm_range(
    np.linspace(
        1e-3,
        1,
        100,
    )
)
df.drop("n", axis=1).plot(loglog=True)
plt.twinx().plot(df["n"], color="black", linestyle="--")

# %%
