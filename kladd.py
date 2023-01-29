# %%
import pennylane as qml
import torch
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=2)

p_01 = np.zeros((4, 4))
p_01[0, 0] = 1
p_01[1, 1] = 1

p_2 = np.zeros((4, 4))
p_2[2, 2] = 1

p_3 = np.zeros((4, 4))
p_3[3, 3] = 1

observables = [
    lambda: qml.probs(wires=[0, 1]),
]


def post_process(obs: torch.Tensor) -> torch.Tensor:
    return obs[..., 0] * obs[..., 1], obs[..., 2], obs[..., 3]


@qml.qnode(dev, interface="torch")
def circuit():

    return [o() for o in observables]


circuit()  # .real
post_process(circuit())
# %%
