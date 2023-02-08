# %%
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from fae import FasterAmplitudeEstimation
from qiskit import Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.algorithms import (  # FasterAmplitudeEstimation,
    AmplitudeEstimation,
    EstimationProblem,
)
from qiskit.primitives import Sampler
from qiskit.utils import QuantumInstance
from scipy import linalg

qi = QuantumInstance(Aer.get_backend("qasm_simulator"))


def mat(size: int) -> np.ndarray:
    diag = (-1) * np.ones(size, dtype=complex)
    diag[0] = 1
    return np.diag(diag)


def s0(reg: QuantumRegister) -> QuantumCircuit:
    # flip phases of all but the 0 state
    # ie apply 2|0><0| - I
    qc = QuantumCircuit(reg)

    qc.unitary(
        mat(size=2 ** len(reg)),
        reg[:],
        label="$S_0$",
    )

    return qc


def ortnormalise(dict_of_rows: dict[int, np.ndarray]) -> np.ndarray:

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

    mat = np.zeros((n_cols, n_cols))
    for i, row in dict_of_rows.items():
        mat[i, :] = row

    return mat.T


class Oracle:
    def __init__(self, p_omega: np.ndarray, rew_func: Callable[[int], float]):
        self.rew_func = rew_func
        self.n_qubits = int(np.ceil(np.log2(len(p_omega))))
        assert len(p_omega) == 2**self.n_qubits
        self.p_omega = p_omega
        self.a_circ = self._build_a_circ()
        self.w_circ = self._build_w_circ()

    def _build_a_circ(self) -> QuantumCircuit:

        mat = ortnormalise({0: np.sqrt(self.p_omega)})

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

        mat = ortnormalise(rows)

        w_reg = QuantumRegister(self.n_qubits, name="w")
        y_reg = QuantumRegister(1, name="y")

        qc = QuantumCircuit(y_reg, w_reg)
        qc.unitary(mat, y_reg[:] + w_reg[:], label="$W$")
        return qc


pw = np.linspace(0.1, 1, 16)
pw /= pw.sum()
good = 15
o = Oracle(pw, lambda x: x == good)


def qmc(o: Oracle, t: int, delta: float) -> float:

    repeats = int(np.log(1 / delta))
    assert repeats > 0
    w_reg = QuantumRegister(o.n_qubits, name="w")
    y_reg = QuantumRegister(1, name="y")

    psi = (
        QuantumCircuit(y_reg, w_reg)
        .compose(o.a_circ, w_reg)
        .compose(o.w_circ, y_reg[:] + w_reg[:])
    )

    grov = QuantumCircuit(y_reg, w_reg)
    grov.z(y_reg)
    grov = grov.compose(psi.inverse(), y_reg[:] + w_reg[:])
    grov = grov.compose(
        s0(y_reg[:] + w_reg[:]),
        y_reg[:] + w_reg[:],
    )
    grov = grov.compose(psi, y_reg[:] + w_reg[:])

    def is_good_state(bitstring: str) -> bool:
        return all(bit == "1" for bit in bitstring)

    problem = EstimationProblem(
        state_preparation=psi,
        objective_qubits=[0],
        grover_operator=grov,
        is_good_state=is_good_state,
    )

    ae = FasterAmplitudeEstimation(
        delta=1e-10,
        # delta=1,
        maxiter=t,
        quantum_instance=qi,
        rescale=False,
        # sampler=sampler,
        shots=t,
    )

    # print(repeats)
    results = [ae.estimate(problem).estimation for _ in range(repeats)]
    print(results)
    # print(ae.estimate(problem).num_oracle_queries)
    # print(ae.estimate(problem).success_probability)
    # print(ae.estimate(problem).num_steps)
    return np.median(results)


for delta in [0.2, 0.1, 0.01, 0.001]:
    val = qmc(o, 1, delta)
    print(delta, val, pw[good])  # , val**2, np.sqrt(val))

# %%
