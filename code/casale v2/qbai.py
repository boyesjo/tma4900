# %%
from typing import Callable

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
        # nu: Callable[[int, int], float],
        # f: Callable[[int, int], bool],
        label: str = "$QBAI$",
        n: int | None = None,
    ):
        self.q_reg = QuantumRegister(x_len + y_len, name="q")
        self.c_reg = ClassicalRegister(x_len, name="c")
        super().__init__(
            self.q_reg,
            self.c_reg,
        )
        self.x_len = x_len
        self.y_len = y_len
        self.len = x_len + y_len
        self.label = label
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
    def ideal_n(p_list: np.ndarray) -> int:
        n = 0.25 * np.pi * np.sqrt(np.mean(p_list) ** (-1)) - 0.5
        return max(round(float(n)), 1)

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

        self.measure(range(self.y_len, self.len), self.c_reg)


# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, execute

x_len = 4
y_len = 2
shots = 10_000

p_list = np.linspace(0.01, 0.1, 2**x_len)
# p_list = np.zeros(2**x_len)
# p_list[0] = 1
# p_list = np.random.rand(2**x_len) * 0.1
# p_list = np.sort(p_list)

random_prob = 1 / np.size(p_list)
expected_prob = (np.max(p_list) / np.mean(p_list)) * random_prob

qc = QBAI(
    x_len,
    y_len,
    p_list,
    n=1,
)

qc.draw("mpl")

# counts = (
#     execute(qc, Aer.get_backend("qasm_simulator"), shots=shots)
#     .result()
#     .get_counts()
# )

# # counts = {int(k, 2): v / shots for k, v in counts.items()}  # int(k[::-1], 2):
# counts_1 = {int(k[::-1], 2): v / shots for k, v in counts.items()}
# plt.title(f"n = {qc.n}")
# plt.bar(counts_1.keys(), counts_1.values())
# plt.axhline(expected_prob, color="red")
# plt.axhline(random_prob, color="green")
# plt.show()

# counts_2 = {int(k, 2): v / shots for k, v in counts.items()}
# plt.title(f"n = {qc.n}")
# plt.bar(counts_2.keys(), counts_2.values())
# plt.axhline(expected_prob, color="red")
# plt.axhline(random_prob, color="green")
# plt.show()


# print(p_list)


# %%
