from typing import Sequence

import pennylane as qml


def u_var(
    n_qubits: int,
    phi: Sequence[float],
    entangle_strat: str = "circular",
):
    assert len(phi) == 2 * n_qubits
    for i in range(n_qubits):
        qml.RX(phi[i], wires=i)
        qml.RY(phi[n_qubits + i], wires=i)

    if entangle_strat == "circular":
        for i in range(n_qubits):
            qml.CZ(wires=[i, (i + 1) % n_qubits])
    elif entangle_strat == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CZ(wires=[i, j])
    elif entangle_strat == "one_to_one":
        for i in range(n_qubits - 1):
            qml.CZ(wires=[i, i + 1])
    else:
        raise ValueError(f"Unknown entangle_strat: {entangle_strat}")


def u_enc(n_qubits: int, s: Sequence[float], lam: Sequence[float]):
    assert len(s) == n_qubits, f"{len(s)} != {n_qubits}, {s}"
    assert len(lam) == 2 * n_qubits

    for i in range(n_qubits):
        qml.RY(lam[i] * s[i], wires=i)
        qml.RZ(lam[n_qubits + i] * s[i], wires=i)


def get_qnn(
    n_qubits: int,
    n_layers: int,
    observables: Sequence,
    entangle_strat: str = "circular",
    device: str = "default.qubit",
):
    dev = qml.device(device, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnn(inputs, phi, lam):

        for i in range(n_qubits):
            qml.Hadamard(wires=i)

        for i in range(n_layers):
            u_var(n_qubits, phi[i])
            u_enc(n_qubits, inputs, lam[i])
        u_var(n_qubits, phi[n_layers])

        return [o() for o in observables]

    return qnn


if __name__ == "__main__":
    from pennylane import numpy as np

    observables = [
        lambda: qml.expval(
            qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3)
        ),
    ]
    test_input = np.array([0.1, 0.2, 0.3, 0.4])
    qnn = get_qnn(4, 2, observables)
    phi = np.random.rand(3, 8)
    lam = np.random.rand(2, 8)
    print(qnn(test_input, phi, lam))
