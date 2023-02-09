# %%
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qbai import ideal_n, qbai
from qiskit import Aer, QuantumCircuit, QuantumRegister, assemble


def get_nu(p_list: np.ndarray, y_len: int = 1) -> Callable[[int, int], float]:
    return (
        lambda x, y: p_list[x]
        if y == 0
        else (1 - p_list[x]) / (2**y_len - 1)
    )


def f(x: int, y: int) -> bool:
    return y == 0


def benchmark(qc: QuantumCircuit, n_shots: int = 1000) -> dict[int, float]:
    counts = (
        Aer.get_backend("qasm_simulator")
        .run(assemble(qc, shots=n_shots))
        .result()
        .get_counts()
    )
    counts = {int(k, 2): v / n_shots for k, v in counts.items()}
    return counts


# %%
def test_ideal_n(
    p_max: np.ndarray,
    n_arms: int = 16,
) -> pd.DataFrame:

    # ideal_n for each p_max
    return pd.DataFrame(
        {
            "p_max": p_max,
            "n": [ideal_n(np.linspace(0.0, p, n_arms)) for p in p_max],
        }
    ).set_index("p_max")


df = test_ideal_n(
    np.linspace(
        0.001,
        1,
        100,
    )
)
df.plot()
plt.yscale("log")
plt.show()


# %%
def test_n(
    n_list: np.ndarray,
    p_list: np.ndarray,
    shots: int = 1000,
) -> pd.DataFrame:

    best_arm = int(np.argmax(p_list))
    n_arms = len(p_list)
    x_len = int(np.log2(n_arms))
    y_len = 2
    x_reg = QuantumRegister(x_len, name="x")
    y_reg = QuantumRegister(y_len, name="y")

    nu = get_nu(p_list, y_len)

    prop_correct_list = []

    for n in n_list:
        qc = qbai(
            x_reg,
            y_reg,
            nu,
            f=f,
            n=n,
        )
        counts = benchmark(qc, shots)
        prop_correct = counts.get(best_arm, 0)
        prop_correct_list.append(prop_correct)

    print(ideal_n(p_list))

    return pd.DataFrame(
        {
            "n": n_list,
            "prop_correct": prop_correct_list,
            "theoretical": np.max(p_list) / np.sum(p_list),
            "random": 1 / n_arms,
        }
    ).set_index("n")


df = test_n(
    n_list=np.arange(0, 20),
    # p_list=np.linspace(0.0, 1, 128),
    p_list=np.linspace(0.0, 0.1, 64),
    shots=10_000,
)

df.plot()


# %%
def test_num_arms(
    n_arms_list: np.ndarray,
    shots: int = 1000,
) -> pd.DataFrame:

    results = []

    for n_arms in n_arms_list:
        p_list = np.linspace(0.0, 0.3, n_arms)
        # p_list = np.random.uniform(0.0, 1, n_arms)
        best_arm = int(np.argmax(p_list))

        x_len = int(np.log2(n_arms))
        y_len = 2
        x_reg = QuantumRegister(x_len, name="x")
        y_reg = QuantumRegister(y_len, name="y")

        nu = get_nu(p_list, y_len)
        n = ideal_n(p_list)
        qc = qbai(
            x_reg,
            y_reg,
            nu,
            f=f,
            n=n,
        )
        counts = benchmark(qc, shots)
        prop_correct = counts.get(best_arm, 0)
        results.append(
            {
                "n_arms": n_arms,
                "prop_correct": prop_correct,
                "n": n,
                "theoerical": np.max(p_list) / np.sum(p_list),
                "random": 1 / n_arms,
            }
        )

    return pd.DataFrame(results).set_index("n_arms")


df = test_num_arms(
    np.array([2**i for i in range(1, 8)]),
    shots=1_000,
)

df[["prop_correct", "theoerical", "random"]].plot()
plt.title(f"n={int(df.n.mean())}")
plt.yscale("log")
plt.xscale("log")
plt.show()


# %%
def test_arms_range(
    max_p: np.ndarray,
    n_arms: int,
    shots: int = 1000,
) -> pd.DataFrame:

    results = []

    for p_range in max_p:
        p_list = np.linspace(0.0, p_range, n_arms)
        best_arm = int(np.argmax(p_list))

        x_len = int(np.log2(n_arms))
        y_len = 1
        x_reg = QuantumRegister(x_len, name="x")
        y_reg = QuantumRegister(y_len, name="y")
        nu = get_nu(p_list, y_len)
        n = ideal_n(p_list)
        qc = qbai(
            x_reg,
            y_reg,
            nu,
            f=f,
            n=n,
        )
        counts = benchmark(qc, shots)
        prop_correct = counts.get(best_arm, 0)
        results.append(
            {
                "max_p": p_range,
                "prop_correct": prop_correct,
                "n": n,
                "theoerical": np.max(p_list) / np.sum(p_list),
                "random": 1 / n_arms,
            }
        )

    return pd.DataFrame(results).set_index("max_p")


df = test_arms_range(
    np.linspace(0.01, 1.0, 40),
    n_arms=64,
    shots=10_000,
)

df[["prop_correct", "theoerical", "random"]].plot()
ax2 = plt.twinx()
ax2.plot(df.index, df.n, color="black", linestyle="dashed", label="n")
ax2.set_yscale("log")
plt.show()
# %%
