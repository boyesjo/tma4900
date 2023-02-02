# %%
from typing import Callable, Optional

import a
import matplotlib.pyplot as plt
import numpy as np
import oracle_e
import oracle_f
import s0
from qiskit import (
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    assemble,
)

# %%
n_arms = 32
x_len = int(np.log2(n_arms))
y_len = 1
all_len = x_len + y_len

# P_LIST = np.linspace(0.1, 0.7, n_arms)
P_LIST = np.random.uniform(0, 1, n_arms) ** 10
best_arm = np.argmax(P_LIST)
prob_correct = np.max(P_LIST) / np.sum(P_LIST)

x_reg = QuantumRegister(x_len, name="x")  # arms register
y_reg = QuantumRegister(y_len, name="y")  # internal state


def nu(x: int, y: int) -> float:
    return P_LIST[x] if y == 1 else 1 - P_LIST[x]


def f(_: int, y: int) -> bool:
    return y == 1


# %%
def qbai(
    x_reg: QuantumRegister,
    y_reg: QuantumRegister,
    nu: Callable[[int, int], float],
    f: Callable[[int, int], bool],
    n: Optional[int] = None,
) -> QuantumCircuit:

    init = (
        QuantumCircuit(y_reg, x_reg)
        .compose(a.circ(x_reg), x_reg)
        .compose(oracle_e.circ(y_reg, x_reg, nu))
    )

    ref = (
        QuantumCircuit(y_reg, x_reg)
        .compose(s0.circ(x_reg), x_reg)
        .compose(s0.circ(y_reg), y_reg)
    )

    grover = (
        QuantumCircuit(y_reg, x_reg)
        .compose(oracle_f.circ(y_reg, x_reg, f))
        .compose(init.inverse(), y_reg[:] + x_reg[:])
        .compose(ref, y_reg[:] + x_reg[:])
        .compose(init, y_reg[:] + x_reg[:])
    )

    c_reg = ClassicalRegister(len(x_reg), name="c")
    qc = QuantumCircuit(y_reg, x_reg, c_reg).compose(init, y_reg[:] + x_reg[:])

    if n is None:
        theta = np.arcsin(np.sqrt(np.mean(P_LIST)))
        n = 0.25 * np.pi / theta - 0.5
        n = max(round(float(n)), 1)

    for _ in range(n):
        qc = qc.compose(grover, y_reg[:] + x_reg[:])

    qc.measure(x_reg, c_reg)
    qc.n = n

    return qc


# %%
SHOTS = 1000

qc = qbai(x_reg, y_reg, nu, f)

counts = (
    Aer.get_backend("qasm_simulator")
    .run(assemble(qc, shots=SHOTS))
    .result()
    .get_counts()
)
counts = {int(k, 2): v / SHOTS for k, v in counts.items()}
colors = ["red" if k == best_arm else "blue" for k in counts.keys()]
plt.title(f"n={qc.n}")
plt.bar(counts.keys(), counts.values(), color=colors)
plt.axhline(prob_correct, color="black", linestyle="--")
plt.axhline(1 / 2**x_len, color="black", linestyle="--")
plt.show()
