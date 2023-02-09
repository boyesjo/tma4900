from typing import Callable

import numpy as np
from complete_unitary import complete_unitary
from qiskit import QuantumCircuit, QuantumRegister


def mat(
    x_len: int,
    y_len: int,
    nu: Callable[[int, int], float],
) -> np.ndarray:

    d = {}

    for x in range(2**x_len):
        idx = x << y_len
        arr = np.zeros(2 ** (x_len + y_len), dtype=complex)

        for y in range(2**y_len):
            arr[idx ^ y] = np.sqrt(nu(x, y))

        d[idx] = arr

    mat = complete_unitary(d)
    return mat.T


def circ(
    y_reg: QuantumRegister,
    x_reg: QuantumRegister,
    nu: Callable[[int, int], float],
    adj: bool = False,
) -> QuantumCircuit:
    qc = QuantumCircuit(y_reg, x_reg)

    u = mat(
        x_len=len(x_reg),
        y_len=len(y_reg),
        nu=nu,
    )

    if adj:
        u = u.conj().T

    qc.unitary(
        u,
        y_reg[:] + x_reg[:],
        label="$O_e^\\dagger$" if adj else "$O_e$",
    )
    return qc


def main() -> None:

    from qiskit import Aer, assemble
    from qiskit.quantum_info import Statevector

    n_arms = 4
    x_len = int(np.log2(n_arms))
    y_len = 2

    x_reg = QuantumRegister(x_len, name="x")
    y_reg = QuantumRegister(y_len, name="y")

    P_LIST = np.linspace(0.1, 0.7, n_arms)

    def nu(x: int, y: int) -> float:
        return P_LIST[x] if y == 0 else (1 - P_LIST[x]) / (2**y_len - 1)

    for arm in range(n_arms):
        qc = QuantumCircuit(y_reg, x_reg)
        # flip bits in x_reg corresponding to 1s in arm binary representation
        for i, bit in enumerate(f"{arm:0b}"[::-1]):
            if bit == "1":
                qc.x(x_reg[i])
        qc = qc.compose(circ(y_reg, x_reg, nu))
        backend = Aer.get_backend("statevector_simulator")
        job = backend.run(assemble(qc))
        result = job.result()
        statevector = Statevector(result.get_statevector())
        probs = statevector.probabilities_dict()
        print(f"arm: {arm}")
        for k, v in probs.items():
            print(f"{k}: {v:.2f}")
        print()


if __name__ == "__main__":
    main()
