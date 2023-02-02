# %%
import numpy as np
import pandas as pd
from qbai import qbai
from qiskit import Aer, QuantumCircuit, QuantumRegister, assemble

# %%
n_arms = 32
x_len = int(np.log2(n_arms))
y_len = 1
x_reg = QuantumRegister(x_len, name="x")  # arms register
y_reg = QuantumRegister(y_len, name="y")  # internal state
all_len = x_len + y_len

P_LIST = np.linspace(0.1, 1, n_arms) / 100


def nu(x: int, y: int) -> float:
    return P_LIST[x] if y == 1 else 1 - P_LIST[x]


def f(_: int, y: int) -> bool:
    return y == 1


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
n_list = np.arange(0, 100)
prop_correct_list = []

for n in n_list:
    qc = qbai(x_reg, y_reg, nu, f, n)
    counts = benchmark(qc, 10000)
    prop_correct = counts[n_arms - 1]
    prop_correct_list.append(prop_correct)


df = pd.DataFrame({"n": n_list, "prop_correct": prop_correct_list}).set_index(
    "n"
)
# %%
