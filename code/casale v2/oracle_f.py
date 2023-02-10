from typing import Callable

import numpy as np
from qiskit import QuantumCircuit


class OracleF(QuantumCircuit):
    def __init__(
        self,
        x_len: int,
        y_len: int,
        f: Callable[[int, int], bool],
        name: str = "$O_f$",
    ):
        super().__init__(x_len + y_len, name=name)
        self.x_len = x_len
        self.y_len = y_len
        self.f = f
        self.name = name
        self._build()

    def _build(self) -> None:
        diag = np.ones(2 ** (self.x_len + self.y_len), dtype=complex)
        for i in range(len(diag)):
            x = i >> self.y_len
            y = i & (2**self.y_len - 1)
            if self.f(x, y):
                diag[i] = -1

        self.unitary(
            np.diag(diag),
            self.qubits,
            label=self.name,
        )

    def adjoint(self) -> QuantumCircuit:
        qc = OracleF(
            x_len=self.x_len,
            y_len=self.y_len,
            f=self.f,
            name=self.name + "$^\\dagger$",
        )
        qc.data = qc.data[::-1]
        return qc

    def inverse(self) -> QuantumCircuit:
        qc = OracleF(
            x_len=self.x_len,
            y_len=self.y_len,
            f=self.f,
        )
        qc.data = qc.data[::-1]
        return qc


if __name__ == "__main__":
    from qiskit import Aer

    x_len = 3
    y_len = 2

    def f(x, y):
        return y == 0

    qc = QuantumCircuit(x_len + y_len)
    qc.h(range(x_len + y_len))
    qc = qc.compose(OracleF(x_len, y_len, f))

    backend = Aer.get_backend("statevector_simulator")
    # job = backend.run(assemble(qc))
    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector(qc)

    for i, amp in enumerate(statevector):
        idx = format(i, f"0{x_len + y_len}b")
        x = int(idx[:x_len], 2)
        y = int(idx[x_len:], 2)
        print(f"{idx} {x} {y} {f(x, y)} {amp}")
