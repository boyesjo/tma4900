from typing import Callable

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def mat(
    x_len: int,
    y_len: int,
    f: Callable[[int, int], bool],
) -> np.ndarray:

    diag = np.ones(2 ** (x_len + y_len), dtype=complex)
    for i in range(2 ** (x_len + y_len)):
        # get bitstring representation of i with leading zeros
        row = format(i, f"0{x_len + y_len}b")
        x = int(row[:x_len], 2)
        y = int(row[x_len:], 2)
        # print(row, x, y)

        if f(x, y):
            diag[i] = -1

    return np.diag(diag)


def circ(
    y_reg: QuantumRegister,
    x_reg: QuantumRegister,
    f: Callable[[int, int], bool],
) -> QuantumCircuit:
    qc = QuantumCircuit(y_reg, x_reg)
    qc.unitary(
        mat(
            x_len=len(x_reg),
            y_len=len(y_reg),
            f=f,
        ),
        y_reg[:] + x_reg[:],
        label="$O_f$",
    )
    return qc


if __name__ == "__main__":
    from qiskit import Aer

    n_arms = 4
    x_len = int(np.log2(n_arms))
    y_len = 1

    x_reg = QuantumRegister(x_len, name="x")
    y_reg = QuantumRegister(y_len, name="y")

    P_LIST = np.linspace(0.1, 0.7, n_arms)

    def f(x, y):
        return y == 1

    qc = QuantumCircuit(y_reg, x_reg)
    qc.h(x_reg)
    qc.h(y_reg)
    qc = qc.compose(circ(y_reg, x_reg, f), [*y_reg, *x_reg])

    backend = Aer.get_backend("statevector_simulator")
    # job = backend.run(assemble(qc))
    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector(qc)

    for i, amp in enumerate(statevector):
        idx = format(i, f"0{x_len + y_len}b")
        x = int(idx[:x_len], 2)
        y = int(idx[x_len:], 2)
        print(f"{idx} {x} {y} {amp}")
