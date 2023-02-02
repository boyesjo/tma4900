import numpy as np
from qiskit import QuantumCircuit, QuantumRegister


def mat(size: int) -> np.ndarray:
    diag = (-1) * np.ones(size, dtype=complex)
    diag[0] = 1
    return np.diag(diag)


def circ(reg: QuantumRegister) -> QuantumCircuit:
    # flip phases of all but the 0 state
    # ie apply 2|0><0| - I
    qc = QuantumCircuit(reg)

    qc.unitary(
        mat(size=2 ** len(reg)),
        reg[:],
        label="$S_0$",
    )

    return qc


if __name__ == "__main__":
    from qiskit import Aer, assemble
    from qiskit.quantum_info import Statevector

    x_len = int(2)
    y_len = 1
    x_reg = QuantumRegister(x_len, name="x")
    y_reg = QuantumRegister(y_len, name="y")

    qc = QuantumCircuit(y_reg, x_reg)
    qc.h(x_reg)
    qc.h(y_reg)
    # qc.x(x_reg)
    # qc.x(y_reg)
    # qc = qc.compose(s0(x_reg), x_reg)
    qc = qc.compose(circ(x_reg), x_reg)
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(assemble(qc))
    result = job.result()
    statevector = Statevector(result.get_statevector())

    # print all states and their probability amplitudes
    for i, amp in enumerate(statevector):
        idx = format(i, f"0{x_len + y_len}b")
        x = int(idx[:x_len], 2)
        y = int(idx[x_len:], 2)
        print(f"{idx} {x} {y} {amp.real}")
