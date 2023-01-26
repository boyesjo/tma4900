# %%
from typing import Sequence

import pennylane as qml
import qnn
import torch
import torch.nn as nn


class RawPQC(nn.Module):
    def __init__(
        self,
        observables: Sequence,
        n_layers: int = 1,
        n_state: int = 2,
        entangle_strat: str = "circular",
        device: str = "default.qubit",
    ):
        super().__init__()
        self.n_actions = len(observables)
        self.n_layers = n_layers
        self.n_qubits = n_state

        self.qnn = qml.qnn.TorchLayer(
            qnn.get_qnn(
                n_qubits=n_state,
                n_layers=n_layers,
                observables=observables,
                entangle_strat=entangle_strat,
                device=device,
            ),
            weight_shapes={
                "phi": (n_layers + 1, 2 * n_state),
                "lam": (n_layers, 2 * n_state),
            },
            init_method={
                "phi": lambda x: nn.init.uniform_(x, 0, 2 * torch.pi),
                "lam": lambda x: nn.init.constant_(x, 1),
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qnn(x)
        return x


# %%
if __name__ == "__main__":
    observables = [
        lambda: qml.expval(
            (qml.Identity(0) + qml.PauliZ(0) @ qml.PauliZ(1)) * 0.5
        ),
        lambda: qml.expval(
            (qml.Identity(0) - qml.PauliZ(0) @ qml.PauliZ(1)) * 0.5
        ),
    ]
    test_input = torch.tensor(
        [
            [0.1, 0.2],
            [0.0, 0.0],
            [0.2, 0.1],
        ]
    )
    model = RawPQC(observables, n_state=2, n_layers=1)
    print(model(test_input))

# %%
