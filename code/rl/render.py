# %%
import gym
import torch
from torch import nn
import pennylane as qml
import time
from typing import Sequence

# %%
env = gym.make("CartPole-v1")
env.reset()


def u_var(n_qubits: int, phi: Sequence[float]):
    assert len(phi) == 2 * n_qubits
    for i in range(n_qubits):
        qml.RX(phi[i], wires=i)
        qml.RY(phi[n_qubits + i], wires=i)

    # entangling layer
    for i in range(n_qubits):
        qml.CZ(wires=[i, (i + 1) % n_qubits])


def u_enc(n_qubits: int, s: Sequence[float], lam: Sequence[float]):
    assert len(s) == n_qubits, f"{len(s)} != {n_qubits}, {s}"
    assert len(lam) == 2 * n_qubits

    for i in range(n_qubits):
        qml.RY(lam[i] * s[i], wires=i)
        qml.RZ(lam[n_qubits + i] * s[i], wires=i)


def get_qnn(n_qubits: int, n_layers: int, device: str = "default.qubit"):

    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnn(inputs, phi, lam):

        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_layers):
            u_var(n_qubits, phi[i])
            u_enc(n_qubits, inputs, lam[i])
        u_var(n_qubits, phi[n_layers])

        # TODO: generalise observables
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    return qnn


class QDQN(nn.Module):
    def __init__(
        self,
        n_obs: int,
        n_actions: int = 2,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.qnn = qml.qnn.TorchLayer(
            get_qnn(n_qubits=n_obs, n_layers=n_layers),
            weight_shapes={
                "phi": (n_layers + 1, 2 * n_obs),
                "lam": (n_layers, 2 * n_obs),
            },
            init_method={
                "phi": lambda x: nn.init.uniform_(x, 0, 2 * torch.pi),
                "lam": lambda x: nn.init.constant_(x, 1),
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qnn(x)
        x = (x + 1) / 2
        x = x.reshape(-1, 1)
        x = torch.cat([x, 1 - x], dim=1)
        return x


QDQN(n_obs=2, n_actions=2)(
    torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.1, 0.2],
        ]
    )
)


# %%
# play and render

model = QDQN(n_obs=4, n_actions=2, n_layers=5)
model.load_state_dict(torch.load("model.pt"))

for _ in range(100):
    state = env.reset()
    done = False
    while not done:
        action = model(torch.tensor(state).float()).argmax().item()
        state, reward, done, _ = env.step(action)
        env.render()
    time.sleep(1)
# %%
