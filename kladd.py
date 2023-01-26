# %%
import pennylane as qml
from pennylane import numpy as np

# %%
dev = qml.device("default.qubit", wires=2)

observables = [
    lambda: qml.expval(qml.PauliZ(0)),
    lambda: qml.expval(qml.PauliZ(1)),
]


@qml.qnode(dev, interface="torch")
def circuit(x, y):
    qml.RX(x, wires=0)
    qml.RX(y, wires=1)

    return [o() for o in observables]


circuit(0, 1).real
# %%
