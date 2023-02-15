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
def ideal_n(p_list: np.ndarray) -> int:
    theta = np.arcsin(np.mean(p_list))
    # theta = np.mean(p_list)
    n = 0.25 * np.pi / theta - 0.5
    return max(round(float(n)), 1)


# %%
def qbai(
    x_reg: QuantumRegister,
    y_reg: QuantumRegister,
    nu: Callable[[int, int], float],
    f: Callable[[int, int], bool],
    n: Optional[int] = None,
    p_list: Optional[np.ndarray] = None,
) -> QuantumCircuit:

    if n is None:
        if p_list is None:
            raise ValueError("Must provide either n or p_list")
        n = ideal_n(p_list)

    all_reg = y_reg[:] + x_reg[:]

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
        .compose(oracle_f.circ(y_reg, x_reg, f), all_reg)
        .compose(init.inverse(), all_reg)
        .compose(ref, all_reg)
        .compose(init, all_reg)
    )

    c_reg = ClassicalRegister(len(x_reg), name="c")
    qc = QuantumCircuit(y_reg, x_reg, c_reg).compose(init, all_reg)

    for _ in range(n):
        qc = qc.compose(grover, all_reg)

    qc.measure(x_reg, c_reg)
    qc.n = n

    return qc


# %%
def main():
    x_len = 4
    n_arms = 2**x_len
    y_len = 2
    SHOTS = 1_000

    P_LIST = np.linspace(0.0, 0.1, n_arms)
    # P_LIST = np.random.uniform(0, 1, n_arms) / 1000
    best_arm = np.argmax(P_LIST)
    prob_correct = np.max(P_LIST) / np.sum(P_LIST)

    x_reg = QuantumRegister(x_len, name="x")  # arms register
    y_reg = QuantumRegister(y_len, name="y")  # internal state

    def nu(x: int, y: int) -> float:
        return P_LIST[x] if y == 0 else (1 - P_LIST[x]) / (y_len**2 - 1)

    def f(_: int, y: int) -> bool:
        return y == 0

    qc = qbai(
        x_reg,
        y_reg,
        nu,
        f,
        n=ideal_n(P_LIST),
        # n = 60,
    )

    qc.draw("mpl")
    plt.show()
    plt.close()

    counts = (
        Aer.get_backend("qasm_simulator")
        .run(assemble(qc, shots=SHOTS))
        .result()
        .get_counts()
    )
    counts = {int(k, 2): v / SHOTS for k, v in counts.items()}

    colors = ["red" if k == best_arm else "blue" for k in counts.keys()]
    plt.title(f"iterations={qc.n}")
    plt.bar(counts.keys(), counts.values(), color=colors, alpha=0.5)
    plt.axhline(prob_correct, color="black", linestyle="--")
    plt.axhline(1 / 2**x_len, color="black", linestyle="--")
    plt.axhline(counts[best_arm], color="red", linestyle="--")
    plt.show()


if __name__ == "__main__":
    main()