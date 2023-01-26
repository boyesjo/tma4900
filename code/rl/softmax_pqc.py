# %%
from typing import Sequence

import pennylane as qml
import qnn
import raw_pqc
import torch
import torch.nn as nn


class SoftmaxPQC(raw_pqc.RawPQC):
    def __init__(
        self,
        observables: Sequence,
        beta: float = 1.0,
        w_length: int = 1,
        n_layers: int = 1,
        n_state: int = 2,
        entangle_strat: str = "circular",
        device: str = "default.qubit",
    ):
        super().__init__(
            observables=observables,
            n_layers=n_layers,
            n_state=n_state,
            entangle_strat=entangle_strat,
            device=device,
        )
        self.beta = beta
        self.w = nn.Parameter(torch.randn(w_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.qnn(x)
        x = self.w * x
        return torch.softmax(self.beta * x, dim=1)


# %%
if __name__ == "__main__":
    observables = [
        lambda: qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        lambda: qml.expval(-1 * qml.PauliZ(0) @ qml.PauliZ(1)),
    ]
    test_input = torch.tensor(
        [
            [0.1, 0.2],
            [0.0, 0.0],
            [0.2, 0.1],
        ]
    )
    model = SoftmaxPQC(observables, n_state=2, n_layers=1, w_length=2)
    print(model(test_input))

# %%
