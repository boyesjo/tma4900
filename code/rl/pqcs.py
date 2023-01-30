# %%
from typing import Callable, Sequence

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
        learnable: bool = False,
        device: str = "default.qubit",
        post_obs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super().__init__()
        self.n_actions = len(observables)
        self.n_layers = n_layers
        self.n_qubits = n_state

        self.post_obs = post_obs

        _qnn, shapes = qnn.get_qnn(
            n_qubits=n_state,
            n_layers=n_layers,
            observables=observables,
            entangle_strat=entangle_strat,
            learnable=learnable,
            device=device,
        )

        init_method = {
            "phi": lambda x: nn.init.uniform_(x, 0, 2 * torch.pi),
            "lam": lambda x: nn.init.constant_(x, 1),
            "theta": lambda x: nn.init.constant_(x, 2 * torch.pi),
        }

        self.qnn = qml.qnn.TorchLayer(
            _qnn,
            weight_shapes=shapes,
            init_method=init_method,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._obs(x)
        return x

    def get_action(self, x: torch.Tensor) -> int:
        return torch.multinomial(self(x), 1).item()  # type: ignore

    def _obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_obs(self.qnn(x))


def test_raw_pqc():
    observables = [
        lambda: qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        # lambda: qml.expval(-1 * qml.PauliZ(0) @ qml.PauliZ(1)),
    ]

    def post_obs(x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return torch.cat([x, 1 - x], dim=1)

    test_input = torch.tensor(
        [
            [0.1, 0.2],
            [0.0, 0.0],
            [0.2, 0.1],
        ]
    )
    model = RawPQC(observables, n_state=2, n_layers=1, post_obs=post_obs)
    print(model(test_input))


class SoftmaxPQC(RawPQC):
    def __init__(
        self,
        observables: Sequence,
        beta: float = 1.0,
        init_w: torch.Tensor = torch.tensor([1.0]),
        n_layers: int = 1,
        n_state: int = 2,
        entangle_strat: str = "circular",
        learnable: bool = False,
        device: str = "default.qubit",
        post_obs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super().__init__(
            observables=observables,
            n_layers=n_layers,
            n_state=n_state,
            entangle_strat=entangle_strat,
            learnable=learnable,
            device=device,
            post_obs=post_obs,
        )
        self.beta = beta
        self.w = nn.Parameter(init_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._obs(x)
        x = self.w * x
        return torch.softmax(self.beta * x, dim=1)


def test_softmax_pqc():

    n_state = 2
    n_layers = 4

    states = torch.rand(3, n_state)

    model = SoftmaxPQC(
        observables=[
            lambda: qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
        ],
        n_state=n_state,
        n_layers=n_layers,
        entangle_strat="one_to_one",
        learnable=True,
        init_w=torch.tensor([1.0]),
        post_obs=lambda x: torch.cat([x, -x], dim=1),
    )
    print(model(states))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))


if __name__ == "__main__":
    test_raw_pqc()
    test_softmax_pqc()

# %%
