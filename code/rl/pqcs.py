# %%
from typing import Callable, Iterable, Sequence

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
        post_obs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super().__init__()
        self.n_actions = len(observables)
        self.n_layers = n_layers
        self.n_qubits = n_state

        self.post_obs = post_obs

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
        x = self._obs(x)
        return x

    def get_action(self, x: torch.Tensor) -> int:
        return torch.multinomial(self(x), 1).item()  # type: ignore

    def _obs(self, x: torch.Tensor) -> torch.Tensor:
        return self.post_obs(self.qnn(x))

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        lr: dict[str, float],
        batch_size: int,
    ) -> None:

        probs = self(states)[range(len(actions)), actions]
        loss = -torch.sum(torch.log(probs) * returns) / batch_size
        loss.backward()

        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in lr:

                    param -= lr[name] * param.grad  # type: ignore
                    param.grad.zero_()


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
        device: str = "default.qubit",
        post_obs: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
    ):
        super().__init__(
            observables=observables,
            n_layers=n_layers,
            n_state=n_state,
            entangle_strat=entangle_strat,
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
    observables = [
        lambda: qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
    ]

    def post_obs(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, -x], dim=1)

    states = torch.tensor(
        [
            [0.1, 0.2],
            [0.0, 0.0],
            [0.2, 0.1],
        ]
    )
    actions = torch.tensor([0, 1, 0])
    test_returns = torch.tensor([1.0, 1.0, 1.0])
    lr = {"qnn.phi": 0.01, "qnn.lam": 0.1, "w": 0.1}
    model = SoftmaxPQC(
        observables,
        n_state=2,
        n_layers=1,
        init_w=torch.tensor([1.0]),
        post_obs=post_obs,
    )
    print(model(states))
    print(list(model.named_parameters()))
    model.update(states, actions, test_returns, lr, batch_size=1)
    print(list(model.named_parameters()))


if __name__ == "__main__":
    test_raw_pqc()
    test_softmax_pqc()

# %%
