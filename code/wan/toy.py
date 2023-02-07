# %%
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms import (
    AmplitudeEstimation,
    EstimationProblem,
    FasterAmplitudeEstimation,
)
from qiskit.utils import QuantumInstance
from scipy import linalg

backend = Aer.get_backend("qasm_simulator")
qi = QuantumInstance(backend)


def ortnormalise(dict_of_rows: dict[int, np.ndarray]) -> dict[int, np.ndarray]:

    n_cols = len(list(dict_of_rows.values())[0])

    # set seed
    np.random.seed(1337)

    for i in range(n_cols):

        if i in dict_of_rows:
            continue

        # new_row = np.zeros(n_cols)
        # new_row[i] = 1

        new_row = np.random.rand(n_cols)

        for row in dict_of_rows.values():
            new_row -= (row @ new_row) * row

        assert linalg.norm(new_row) > 1e-10
        new_row /= linalg.norm(new_row)
        dict_of_rows[i] = new_row

    return dict_of_rows


class Oracle:
    def __init__(self, p_omega: np.ndarray, rew_func: Callable[[int], float]):
        self.rew_func = rew_func
        self.n_qubits = int(np.ceil(np.log2(len(p_omega))))
        assert len(p_omega) == 2**self.n_qubits
        self.p_omega = p_omega
        self.a_circ = self._build_a_circ()
        self.w_circ = self._build_w_circ()

    def _build_a_circ(self) -> QuantumCircuit:

        rows = ortnormalise({0: np.sqrt(self.p_omega)})
        mat = np.array(list(rows.values())).T

        q_reg = QuantumRegister(self.n_qubits, name="x")
        qc = QuantumCircuit(q_reg)
        qc.unitary(mat, q_reg[:], label="$O$")
        return qc

    def _build_w_circ(self) -> QuantumCircuit:

        rows: dict[int, np.ndarray] = {}
        for omega in range(len(self.p_omega)):
            row = np.zeros(2 ** (self.n_qubits + 1))
            row[omega << 1] = np.sqrt(1 - self.rew_func(omega))
            row[(omega << 1) + 1] = np.sqrt(self.rew_func(omega))

            rows[omega << 1] = row

        rows = ortnormalise(rows)
        mat = np.array(list(rows.values())).T

        w_reg = QuantumRegister(self.n_qubits, name="w")
        y_reg = QuantumRegister(1, name="y")

        qc = QuantumCircuit(y_reg, w_reg)
        qc.unitary(mat, y_reg[:] + w_reg[:], label="$W$")
        return qc


pw = np.linspace(0.1, 1, 4)
pw /= pw.sum()
o = Oracle(pw, lambda x: x == 4)


def qmc(o: Oracle, t: int, delta: float) -> float:

    shots = int(np.log(1 / delta))
    w_reg = QuantumRegister(o.n_qubits, name="w")
    y_reg = QuantumRegister(1, name="y")
    c_reg = ClassicalRegister(1, name="c")

    psi = (
        QuantumCircuit(y_reg, w_reg)
        .compose(o.a_circ, w_reg)
        .compose(o.w_circ, y_reg[:] + w_reg[:])
    )

    proj = QuantumCircuit(y_reg, w_reg)
    proj.z(y_reg)
    # proj = proj.compose(psi, y_reg[:] + w_reg[:])
    # # reflect about |1>
    # proj.x(y_reg)
    # proj.h(y_reg)
    # proj.append(psi.to_gate(), y_reg[:] + w_reg[:])
    # proj.h(y_reg)
    # proj.x(y_reg)

    problem = EstimationProblem(
        state_preparation=psi,
        objective_qubits=0,
        grover_operator=proj,
        # is_good_state=lambda x: x == 1,
    )

    ae = FasterAmplitudeEstimation(
        delta=1e-20,
        maxiter=t,
        quantum_instance=qi,
        # sampler=shots,
    )

    result = ae.estimate(problem)
    return result.estimation


for t in range(1, 7):
    print(t, qmc(o, t, 0.001))


# %%


def oracle(prob: float, seed: int = 1337) -> QuantumCircuit:

    rows = {
        0: np.array([np.sqrt(1 - prob), 0, 0, np.sqrt(prob)]),
    }

    np.random.seed(seed)

    # add arbitrary rows, ensuring ortonormality
    for i in range(1, 2**2):
        # new_row = np.random.rand(2**2)
        new_row = np.zeros(2**2)
        new_row[i] = 1
        for row in rows.values():
            new_row -= (row @ new_row) * row
        assert linalg.norm(new_row) > 1e-10
        new_row /= linalg.norm(new_row)
        rows[i] = new_row

    qc = QuantumCircuit(2)
    qc.unitary(np.array(list(rows.values())).T, [0, 1], label="$O_f$")

    return qc


qc = QuantumCircuit(2)
qc.x([0, 1])
qc = qc.compose(oracle(0.1))
state = (
    Aer.get_backend("statevector_simulator").run(qc).result().get_statevector()
)
state.draw("latex")


# %%
p_list = np.linspace(0.1, 0.7, 4)
oracle_list = [oracle(p) for p in p_list]


def phi(x: int) -> float:
    return x & 1


def w(phi: Callable[[int], float]) -> QuantumCircuit:
    # map |x>|0> to |x>|1> if x = 11, else to |x>|0>
    x_len = 2
    mat = np.zeros((2 ** (x_len + 1), 2 ** (x_len + 1)))
    for row in range(2 ** (x_len + 1)):
        x = row >> 1
        y = row & 1
        if y == 0:
            mat[row, row] = np.sqrt(1 - phi(x))
            mat[row, row + 1] = np.sqrt(phi(x))
        else:
            mat[row, row - 1] = np.sqrt(phi(x))
            mat[row, row] = -np.sqrt(1 - phi(x))

    x_reg = QuantumRegister(x_len, name="x")
    y_reg = QuantumRegister(1, name="y")

    qc = QuantumCircuit(y_reg, x_reg)
    qc.unitary(mat.T, [*y_reg, *x_reg], label="$W$")
    return qc


x_reg = QuantumRegister(2, name="x")
y_reg = QuantumRegister(1, name="y")
qc = QuantumCircuit(y_reg, x_reg)
qc.x(x_reg)
qc = qc.compose(w(phi))
state = (
    Aer.get_backend("statevector_simulator").run(qc).result().get_statevector()
)
state.draw("latex")


# %%
calls = 0


def qmc(alg: QuantumCircuit, r: float, delta: float) -> QuantumCircuit:
    n_iter = int(np.ceil(np.log2(1 / delta)))
    results = np.zeros(n_iter)

    shots = 1000
    t = 1
    calls = 0

    x_reg = QuantumRegister(len(alg.qubits), name="x")
    y_reg = QuantumRegister(1, name="y")
    c_reg = ClassicalRegister(1, name="c")

    qc = QuantumCircuit(y_reg, x_reg, c_reg)

    for _ in range(t):
        qc = qc.compose(alg, x_reg)
        qc = qc.compose(w(phi), [*y_reg, *x_reg])
    qc.measure(y_reg, c_reg)

    for i in range(n_iter):
        counts = (
            Aer.get_backend("qasm_simulator")
            .run(qc, shots=shots)
            .result()
            .get_counts()
        )
        results[i] = counts.get("1", 0) / shots

    return np.median(results)


qmc(oracle_list[0], 0.1, 0.1)
# %%
