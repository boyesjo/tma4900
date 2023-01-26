# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np
from qiskit import (
    IBMQ,
    Aer,
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    assemble,
    transpile,
)
from qiskit.algorithms import AmplificationProblem, AmplitudeAmplifier, Grover
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator, Statevector

warnings.filterwarnings("ignore", category=DeprecationWarning)

# %%
n_arms = 2**4
x_len = int(np.log2(n_arms))

P_LIST = np.linspace(0, 1, n_arms)


def nu_x(x: int, y: int) -> float:
    return P_LIST[x] if y == 1 else 1 - P_LIST[x]


def f(x: int, y: int) -> bool:
    # return (y + x) % 2 == 1
    return y == 1


y_len = 1
x_reg = QuantumRegister(x_len, name="x")
y_reg = QuantumRegister(y_len, name="y")
# all_qubits = [*y_reg, *x_reg]
all_qubits = range(x_len + y_len)
c_reg = ClassicalRegister(x_len, name="c")


# %%
def show_matrix(matrix: np.ndarray) -> None:
    size = len(x_reg) + len(y_reg)
    plt.matshow(matrix.real)
    # binary labels for x and y
    labels = [format(i, f"0{size}b") for i in range(2**size)]
    # plt.xticks(range(2**size - 1, -1, -1), labels)
    # plt.yticks(range(2**size - 1, -1, -1), labels)
    plt.xticks(range(2**size), labels)
    plt.yticks(range(2**size), labels)
    # show large
    plt.gcf().set_size_inches(10, 10)
    plt.colorbar()
    plt.show()


# %%
def oracle_e_matrix() -> QuantumCircuit:
    # maps |x>|0> to |x>|nu(x)>
    n_qubits = len(x_reg) + len(y_reg)
    n_states = 2**n_qubits

    matrix = np.zeros((n_states, n_states), dtype=complex)
    for i in range(n_states):
        # get bitstring representation of i with leading zeros
        row = format(i, f"0{n_qubits}b")
        x = int(row[:x_len], 2)
        y = int(row[x_len:], 2)

        # print(row, x, y)

        new_row = np.zeros(n_states, dtype=complex)
        # loop through possible y values
        for new_y in range(2**y_len):
            # TODO: ensure correctness for y_len > 1
            col = x << y_len | new_y ^ y

            # print(x, new_y, np.sqrt(nu_x(x, new_y)))

            val = np.sqrt(nu_x(x, new_y))
            new_row[col] = val

            # flip sign of amplitudes corresponding to ones in new_y bitstring
            # TODO: ensure correctness for y_len > 1
            for j in range(y_len):
                if y & (1 << j):
                    new_row[x << y_len ^ j] *= -1

        # print(row, new_row)

        # print(
        #     x,
        #     y,
        #     {key: val.real for key, val in enumerate(new_row) if val != 0},
        # )

        matrix[i] = new_row

    # reorder rows and columns to match qiskit convention
    # matrix = matrix[::-1, ::-1]

    # show_matrix(matrix)
    # assert unitarity
    assert np.allclose(matrix @ matrix.conj().T, np.eye(n_states))
    return matrix.T


def oracle_e() -> QuantumCircuit:

    matrix = oracle_e_matrix()
    qc = QuantumCircuit(y_reg, x_reg)
    qc.unitary(matrix, all_qubits, label="$O_e$")

    return qc


def oracle_e_adj() -> QuantumCircuit:
    matrix = oracle_e_matrix().conj().T
    qc = QuantumCircuit(x_reg, y_reg)
    qc.unitary(matrix, all_qubits, label="$O_e^\\dagger$")

    return qc


qc = QuantumCircuit(y_reg, x_reg)
# qc.x(x_reg)
# qc.x(y_reg)
qc = qc.compose(oracle_e())
# qc += oracle_e()
backend = Aer.get_backend("statevector_simulator")
job = backend.run(assemble(qc))
result = job.result()
statevector = Statevector(result.get_statevector())
statevector.draw("latex")
# probs = np.abs(statevector.data) ** 2
# probs

# %%
def a_op() -> QuantumCircuit:
    qc = QuantumCircuit(x_reg)
    qc.h(x_reg)
    return qc


# %%
def oracle_f() -> QuantumCircuit:
    # maps |x>|y> to (-1)^f(x, y) |x>|y>

    diag = np.ones(2 ** (x_len + y_len), dtype=complex)
    for i in range(2 ** (x_len + y_len)):
        # get bitstring representation of i with leading zeros
        row = format(i, f"0{x_len + y_len}b")
        x = int(row[:x_len], 2)
        y = int(row[x_len:], 2)
        # print(row, x, y)

        if f(x, y):
            diag[i] = -1

    # diag = diag[::-1]

    qc = QuantumCircuit(x_reg, y_reg)
    qc.unitary(np.diag(diag), all_qubits, label="$O_f$")
    # qc.unitary(np.diag(diag), all_qubits, label="$O_f$")
    return qc


qc = QuantumCircuit(y_reg, x_reg)
# qc.x(x_reg[0])
# qc.x(y_reg)
qc.h(all_qubits)
qc = qc.compose(oracle_f())
backend = Aer.get_backend("statevector_simulator")
job = backend.run(assemble(qc))
result = job.result()
statevector = Statevector(result.get_statevector())
statevector.draw("latex")
# probs = np.abs(statevector.data) ** 2
# probs


# %%
def s0(reg) -> QuantumCircuit:
    # reflection about 0 state
    qc = QuantumCircuit(reg)
    # ket_0 = np.zeros(2 ** len(reg), dtype=complex)
    # ket_0[0] = 1
    # matrix = np.eye(2 ** len(reg)) - 2 * np.outer(ket_0, ket_0)
    diag = (-1) * np.ones(2 ** len(reg), dtype=complex)
    diag[0] = 1
    qc.unitary(np.diag(diag), reg, label="$S_0$")
    return qc


qc = QuantumCircuit(y_reg, x_reg)
qc.h(all_qubits)
# qc.x(x_reg)
# qc.x(y_reg)
# qc = qc.compose(s0(x_reg), x_reg)
qc = qc.compose(s0(x_reg), x_reg)
backend = Aer.get_backend("statevector_simulator")
job = backend.run(assemble(qc))
result = job.result()
statevector = Statevector(result.get_statevector())
statevector.draw("latex")
# probs = np.abs(statevector.data) ** 2
# probs


# %%
def qbai(n):
    qc = QuantumCircuit(y_reg, x_reg, c_reg)
    qc += a_op()
    qc += oracle_e()

    for _ in range(n):
        qc = qc.compose(oracle_f(), all_qubits)
        qc = qc.compose(oracle_e_adj(), all_qubits)
        qc = qc.compose(a_op(), x_reg)
        qc = qc.compose(s0(x_reg), x_reg)
        qc = qc.compose(s0(y_reg), y_reg)
        qc = qc.compose(a_op(), x_reg)
        qc = qc.compose(oracle_e(), all_qubits)

    qc.measure(x_reg, c_reg)

    return qc


qbai(2).draw("mpl")

# %%
n = 1
backed = Aer.get_backend("qasm_simulator")
qc = qbai(100)
job = backed.run(qc, shots=1000)
result = job.result()

# plot results of measurement with int as x-axis
counts = result.get_counts()
counts = {int(k, 2): v for k, v in counts.items()}
plt.bar(counts.keys(), counts.values())
plt.show()

# %%
