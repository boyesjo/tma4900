# %%
import matplotlib.pyplot as plt
import numpy as np
from qiskit import Aer, QuantumCircuit, execute

shots = 1000

qc = QuantumCircuit(2)
qc.x([0, 1])
qc.unitary(
    np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, np.sqrt(0.5), 0, np.sqrt(0.5)],
            [0, -np.sqrt(0.5), 0, np.sqrt(0.5)],
        ]
    ).T,
    range(2),
)
qc.measure_all()

counts = (
    execute(qc, Aer.get_backend("qasm_simulator"), shots=shots)
    .result()
    .get_counts()
)

# counts = {int(k, 2): v / shots for k, v in counts.items()}  # int(k[::-1], 2):
counts = {int(k, 2): v / shots for k, v in counts.items()}

plt.bar(counts.keys(), counts.values())

# %%
