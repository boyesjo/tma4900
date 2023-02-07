# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, QuantumCircuit, QuantumRegister


# %%
def oracle(prob: float) -> QuantumCircuit:
    mat = np.array(
        [
            [np.sqrt(1 - prob), 0, 0, np.sqrt(prob)],
            [-np.sqrt(prob), 0, 0, np.sqrt(1 - prob)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    qc = QuantumCircuit(2)

    qc.unitary(mat.T, [0, 1])
    return qc


qc = oracle(0.1)
state = (
    Aer.get_backend("statevector_simulator").run(qc).result().get_statevector()
)

# %%
p_list = np.linspace(0.1, 0.7, 4)
oracle_list = [oracle(p) for p in p_list]


# %%
def qmc(qc: QuantumCircuit, r: float, delta: float) -> QuantumCircuit:
    pass
