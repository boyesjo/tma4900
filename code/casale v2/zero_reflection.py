import numpy as np
from qiskit import QuantumCircuit


class ZeroReflection(QuantumCircuit):
    def __init__(self, n: int, label: str = "$S_0$"):
        super().__init__(n)
        self.n = n
        self.label = label
        self._build()

    def _build(self) -> None:
        diag = np.ones(2**self.n, dtype=complex)
        diag[0] = -1
        self.unitary(
            np.diag(diag),
            self.qubits,
            label=self.label,
        )

    def adjoint(self) -> QuantumCircuit:
        qc = ZeroReflection(n=self.n, label=self.label)
        qc.data = qc.data[::-1]
        return qc

    def inverse(self) -> QuantumCircuit:
        qc = ZeroReflection(n=self.n, label=self.label)
        qc.data = qc.data[::-1]
        return qc


if __name__ == "__main__":
    from qiskit import Aer, assemble
    from qiskit.quantum_info import Statevector

    n = 4
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc = qc.compose(ZeroReflection(4))
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(assemble(qc))
    result = job.result()
    statevector = Statevector(result.get_statevector())

    # print all states and their probability amplitudes
    for i, amp in enumerate(statevector):
        idx = format(i, f"0{4}b")
        print(f"{idx} {amp.real}")
