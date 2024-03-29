from typing import Optional, Sequence

import pennylane as qml


def theta_shape(n_qubits: int, entangle_strat: str) -> int:
    if entangle_strat == "one_to_one":
        return n_qubits - 1
    elif entangle_strat == "circular":
        return n_qubits
    elif entangle_strat == "all_to_all":
        return n_qubits * (n_qubits - 1) // 2
    else:
        raise ValueError(f"Unknown entangle_strat: {entangle_strat}")


def r_zz(theta: float, wires: Sequence[int]) -> None:
    assert len(wires) == 2
    qml.CNOT(wires=wires)
    qml.RZ(theta, wires=wires[1])
    qml.CNOT(wires=wires)


def cz_wrap(_: float, wires: Sequence[int]) -> None:
    assert len(wires) == 2
    qml.CZ(wires=wires)


def entangle(
    n_qubits: int,
    theta: Optional[Sequence[float]] = None,
    learnable: bool = False,
    entangle_strat: str = "circular",
) -> None:

    func = r_zz if learnable else cz_wrap
    if theta is None:
        theta = [0.0] * n_qubits**2

    # assert len(theta) == theta_shape(n_qubits, entangle_strat)

    if entangle_strat == "one_to_one":
        for i in range(n_qubits - 1):
            func(theta[i], wires=[i, i + 1])
    elif entangle_strat == "circular":
        for i in range(n_qubits):
            func(theta[i], wires=[i, (i + 1) % n_qubits])
    elif entangle_strat == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                func(theta[i], wires=[i, j])
    else:
        raise ValueError(f"Unknown entangle_strat: {entangle_strat}")


def u_var(
    n_qubits: int,
    phi: Sequence[float],
    theta: Optional[Sequence[float]] = None,
    learnable: bool = False,
    entangle_strat: str = "circular",
):
    assert len(phi) == 2 * n_qubits
    for i in range(n_qubits):
        qml.RZ(phi[i], wires=i)
        qml.RY(phi[n_qubits + i], wires=i)

    entangle(n_qubits, theta, learnable, entangle_strat)


def u_enc(n_qubits: int, s: Sequence[float], lam: Sequence[float]):
    assert len(s) == n_qubits, f"{len(s)} != {n_qubits}, {s}"
    assert len(lam) == 2 * n_qubits

    for i in range(n_qubits):
        qml.RY(lam[i] * s[i], wires=i)
        qml.RZ(lam[n_qubits + i] * s[i], wires=i)


def get_unlearnable_qnn(
    n_qubits: int,
    n_layers: int,
    observables: Sequence,
    entangle_strat: str = "circular",
    device: str = "default.qubit",
) -> tuple[qml.QNode, dict[str, tuple[int, int]]]:
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnn(inputs, phi, lam):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_layers):
            u_var(
                n_qubits,
                phi[i],
                learnable=False,
                entangle_strat=entangle_strat,
            )
            u_enc(n_qubits, inputs, lam[i])
        u_var(
            n_qubits,
            phi[-1],
            learnable=False,
            entangle_strat=entangle_strat,
        )

        return [o() for o in observables]

    weight_shapes = {
        "phi": (n_layers + 1, 2 * n_qubits),
        "lam": (n_layers, 2 * n_qubits),
    }

    return qnn, weight_shapes


def get_learnable_qnn(
    n_qubits: int,
    n_layers: int,
    observables: Sequence,
    entangle_strat: str = "circular",
    device: str = "default.qubit",
) -> tuple[qml.QNode, dict[str, tuple[int, int]]]:
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnn(inputs, phi, lam, theta):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_layers):
            u_var(
                n_qubits,
                phi[i],
                theta[i],
                learnable=True,
                entangle_strat=entangle_strat,
            )
            u_enc(n_qubits, inputs, lam[i])
        u_var(n_qubits, phi[-1], learnable=True, entangle_strat=entangle_strat)

        return [o() for o in observables]

    weight_shapes = {
        "phi": (n_layers + 1, 2 * n_qubits),
        "lam": (n_layers, 2 * n_qubits),
        "theta": (n_layers + 1, theta_shape(n_qubits, entangle_strat)),
    }

    return qnn, weight_shapes


def get_qnn(
    n_qubits: int,
    n_layers: int,
    observables: Sequence,
    entangle_strat: str = "circular",
    learnable: bool = False,
    device: str = "default.qubit",
) -> tuple[qml.QNode, dict[str, tuple[int, ...]]]:

    if learnable:
        qnn, weight_shapes = get_learnable_qnn(
            n_qubits,
            n_layers,
            observables,
            entangle_strat,
            device,
        )
    else:
        qnn, weight_shapes = get_unlearnable_qnn(
            n_qubits,
            n_layers,
            observables,
            entangle_strat,
            device,
        )

    return qnn, weight_shapes  # type: ignore


if __name__ == "__main__":
    from pennylane import numpy as np

    observables = [
        lambda: qml.expval(
            qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
        ),
    ]
    test_input = np.array([0.1, 0.2, 0.3, 0.4])
    qnn, shapes = get_qnn(4, 2, observables, learnable=True)
    phi = np.random.rand(3, 8)
    lam = np.random.rand(2, 8)
    theta = np.random.rand(3, 4)
    print(qnn(test_input, phi, lam, theta))
    print(shapes)
