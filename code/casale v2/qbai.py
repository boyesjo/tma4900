import numpy as np
from oracle_e import OracleE
from oracle_f import OracleF
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from zero_reflection import ZeroReflection


class QBAI(QuantumCircuit):
    def __init__(
        self,
        x_len: int,
        y_len: int,
        p_list: np.ndarray,
        name: str = "$QBAI$",
        n: int | None = None,
    ):
        self.q_reg = QuantumRegister(x_len + y_len, name="q")
        self.c_reg = ClassicalRegister(x_len, name="c")
        super().__init__(
            self.q_reg,
            name=name,
        )

        self.x_len = x_len
        self.y_len = y_len
        self.len = x_len + y_len
        self.name = name
        self.p_list = p_list

        self.nu = (
            lambda x, y: p_list[x]
            if y == 0
            else (1 - p_list[x]) / (2**y_len - 1)
        )

        self.f = lambda x, y: y == 0

        self.o_e = OracleE(x_len, y_len, self.nu)
        self.o_f = OracleF(x_len, y_len, self.f)
        self.s0x = ZeroReflection(x_len)
        self.s0y = ZeroReflection(y_len)

        if n is None:
            n = self.ideal_n(p_list)
        self._build(n)

    @staticmethod
    def ideal_n_exact(p_list: np.ndarray) -> float:
        # theta = np.arcsin(np.sqrt(np.mean(p_list)))
        theta = np.sqrt(np.max(p_list))
        n = 0.25 * np.pi / theta - 0.5
        return float(n)

    @staticmethod
    def ideal_n(p_list: np.ndarray) -> int:
        return max(round(QBAI.ideal_n_exact(p_list)), 1)

    def _build(self, n: int) -> None:
        self.n = n
        self.h(range(self.y_len, self.len))
        self.append(self.o_e, range(self.len))

        for _ in range(n):
            self.append(self.o_f, range(self.len))

            self.append(self.o_e.adjoint(), range(self.len))
            self.h(range(self.y_len, self.len))

            self.barrier()
            self.append(self.s0x, range(self.y_len, self.len))
            self.append(self.s0y, range(self.y_len))
            self.barrier()

            self.h(range(self.y_len, self.len))
            self.append(self.o_e, range(self.len))

    def add_measurements(self) -> None:
        self.add_register(self.c_reg)
        self.measure(range(self.y_len, self.len), self.c_reg)


def main() -> None:
    import matplotlib.pyplot as plt
    from qiskit import Aer, execute

    x_len = 4
    y_len = 2
    shots = 10_000

    # p_list = np.linspace(0.01, 0.02, 2**x_len)

    p_list = np.zeros(2**x_len)
    p_list = np.ones(2**x_len) * 0.1
    p_list[10] = 1

    # p_list = np.random.rand(2**x_len) * 0.01
    # p_list = np.sort(p_list)

    random_prob = 1 / np.size(p_list)
    expected_prob = (np.max(p_list) / np.mean(p_list)) * random_prob

    qc = QBAI(
        x_len,
        y_len,
        p_list,
    )
    qc.add_measurements()

    counts = (
        execute(qc, Aer.get_backend("qasm_simulator"), shots=shots)
        .result()
        .get_counts()
    )

    counts = {int(k, 2): v / shots for k, v in counts.items()}
    plt.title(f"n = {qc.n}")
    plt.bar(counts.keys(), counts.values())
    plt.axhline(expected_prob, color="red")
    plt.axhline(random_prob, color="green")
    plt.show()

    print(p_list)

    QBAI(4, 2, np.linspace(0.01, 0.1, 2**4), n=2).draw("mpl")
    plt.show()


if __name__ == "__main__":
    main()
