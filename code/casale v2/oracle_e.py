from typing import Callable

import numpy as np
from complete_unitary import complete_unitary
from qiskit import QuantumCircuit


class OracleE(QuantumCircuit):
    def __init__(
        self,
        x_len: int,
        y_len: int,
        nu: Callable[[int, int], float],
        name: str = "$O_e$",
        adj: bool = False,
    ):
        super().__init__(x_len + y_len, name=name)
        self.x_len = x_len
        self.y_len = y_len
        self.nu = nu
        self.name = name
        self.adj = adj
        self._build()

    def _matrix(self) -> np.ndarray:
        d = {}

        for x in range(2**self.x_len):
            idx = x << self.y_len
            arr = np.zeros(2 ** (self.x_len + self.y_len), dtype=complex)

            for y in range(2**self.y_len):
                arr[idx ^ y] = np.sqrt(self.nu(x, y))

            d[idx] = arr

        return complete_unitary(d).T

    def _build(self) -> None:
        mat = self._matrix()

        if self.adj is True:
            mat = mat.conj().T

        self.unitary(
            mat,
            self.qubits,
            label=self.name,
        )

    def adjoint(self) -> QuantumCircuit:
        qc = OracleE(
            x_len=self.x_len,
            y_len=self.y_len,
            nu=self.nu,
            name=self.name + "$^\\dagger$",
            adj=not self.adj,
        )
        qc.data = qc.data[::-1]
        return qc


def main() -> None:

    from qiskit import Aer, assemble
    from qiskit.quantum_info import Statevector

    n_arms = 4
    x_len = int(np.log2(n_arms))
    y_len = 2

    P_LIST = np.linspace(0.1, 0.7, n_arms)

    def nu(x: int, y: int) -> float:
        return P_LIST[x] if y == 0 else (1 - P_LIST[x]) / (2**y_len - 1)

    qc = QuantumCircuit(y_len + x_len)
    # qc.x([3, 2])
    qc = qc.compose(OracleE(x_len, y_len, nu))

    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(assemble(qc))
    result = job.result()
    statevector = Statevector(result.get_statevector())
    probs = statevector.probabilities_dict()
    for k, v in probs.items():
        print(f"{k}: {v:.2f}")
    print()

    import matplotlib.pyplot as plt

    qc.draw("mpl")
    plt.show()


if __name__ == "__main__":
    main()
