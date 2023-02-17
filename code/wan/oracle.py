import numpy as np
from complete_unitary import complete_unitary
from qiskit import QuantumCircuit, QuantumRegister


class Oracle(QuantumCircuit):
    def __init__(
        self,
        p: float,
        w_len: int,
        name: str = "$O_x$",
    ):
        self.p_ = p
        self.w_len = w_len
        self.w_reg = QuantumRegister(w_len, name="w")
        self.y_reg = QuantumRegister(1, name="y")
        super().__init__(self.y_reg, self.w_reg, name=name)
        self._build()

    def _matrix(self) -> np.ndarray:
        p_w = np.zeros(2**self.w_len)

        def y_x(w: int) -> int:
            return 1 if w == 0 else 0

        p_w[0] = self.p_
        p_w[1:] = (1 - self.p_) / (2**self.w_len - 1)

        row = np.zeros(2 ** (self.w_len + 1))

        for w in range(2**self.w_len):

            for y in range(2):
                col = 2 * w + y
                row[col] = np.sqrt(p_w[w]) if y == y_x(w) else 0

        return complete_unitary({0: row}).T

    def _build(self) -> None:
        mat = self._matrix()

        self.unitary(
            mat,
            self.qubits,
            label=self.name,
        )

    def inverse(self) -> QuantumCircuit:
        inv = super().inverse()
        inv.name = self.name + "$^\\dagger$"
        return inv


def main():
    from qiskit import Aer, assemble
    from qiskit.quantum_info import Statevector

    qc = Oracle(0.7, 2)
    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(assemble(qc))
    result = job.result()
    statevector = Statevector(result.get_statevector())
    probs = statevector.probabilities_dict()
    for k, v in probs.items():
        print(f"{k}: {v:.3f}")
    print()


if __name__ == "__main__":
    main()
