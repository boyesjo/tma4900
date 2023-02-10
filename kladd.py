# %%
import numpy as np
from qiskit import QuantumCircuit

qc = QuantumCircuit(2)
qc.unitary(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, np.sqrt(0.5), 0, np.sqrt(0.5)],
        [0, -np.sqrt(0.5), 0, np.sqrt(0.5)],
    ]
)
