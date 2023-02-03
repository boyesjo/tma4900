from typing import Callable, Optional

import a
import matplotlib.pyplot as plt
import numpy as np
import oracle_f
import s0
from qiskit import QuantumCircuit, QuantumRegister


def grover(
    n_qubits: int,
    f: Callable[[int, int], bool],
    n: Optional[int] = None,
    n_correct: int = 1,
) -> QuantumCircuit:

    q_reg = QuantumRegister(n_qubits, name="q")
    o = QuantumCircuit(q_reg).compose(
        oracle_f.circ(x_reg=q_reg, y_reg=[], f=f), q_reg
    )

    r = (
        QuantumCircuit(q_reg)
        .compose(a.circ(q_reg), q_reg)
        .compose(s0.circ(q_reg), q_reg)
        .compose(a.circ(q_reg), q_reg)
    )

    if n is None:
        n = round(0.25 * np.pi * np.sqrt(2**n_qubits / n_correct))

    qc = QuantumCircuit(q_reg)
    qc.h(q_reg)
    for _ in range(n):
        qc = qc.compose(o, q_reg).compose(r, q_reg)

    qc.measure_all()
    qc.n = n
    return qc


def main():
    from qiskit import Aer, assemble

    n_qubits = 7
    shots = 10_000

    def f(x: int, _: int = 0) -> bool:
        return x in (3, 5, 6, 7, 8)

    qc = grover(n_qubits, f, n_correct=5)
    counts = (
        Aer.get_backend("qasm_simulator")
        .run(assemble(qc, shots=shots))
        .result()
        .get_counts()
    )
    counts = {int(k, 2): v / shots for k, v in counts.items()}
    colors = ["red" if f(x) else "blue" for x in counts.keys()]
    plt.bar(counts.keys(), counts.values(), color=colors)
    plt.title(f"n={qc.n}")
    plt.axhline(1 / 2**n_qubits, color="black", linestyle="--")
    plt.xlim(-1, 2**n_qubits)
    plt.show()


if __name__ == "__main__":
    main()
