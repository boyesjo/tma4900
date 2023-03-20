# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power, kron


def rotation_matrix(n_qubits: int, state: int):
    """
    Rotation matrix for a given state.
    Corresponding to one Grover iteration.
    """

    assert 0 <= state < 2**n_qubits

    oracle = np.ones(2**n_qubits, dtype=complex)
    oracle[state] = -1
    oracle = np.diag(oracle)

    # matrix equivalent hadamarding all qubits
    h_gate = np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)
    h = np.eye(1, dtype=complex)
    # h = h_gate.copy()
    for _ in range(n_qubits):
        h = kron(h, h_gate)

    zero_refl = np.ones(2**n_qubits, dtype=complex)
    zero_refl[0] = -1
    zero_refl = np.diag(zero_refl)

    # one grover iteration
    return h @ zero_refl @ h @ oracle


rotation_matrix(2, 0)


def sample(state_vector: np.ndarray, n_samples: int = 1):
    return np.random.choice(
        len(state_vector),
        size=n_samples,
        p=np.abs(state_vector) ** 2,
    )


# %%
N_QUBITS = 5
P_LIST = np.linspace(0.0, 1.0, 2**N_QUBITS)
# P_LIST = np.array([i == 0 for i in range(2**N_QUBITS)], dtype=float)
HORIZON = 1000

SUCC_ANGLE = 0.5
FAIL_ANGLE = -0.1

regrets = np.zeros(HORIZON)
arms = np.zeros(HORIZON, dtype=int)
succs = np.zeros(HORIZON, dtype=int)

SUCC_ROTS = [
    fractional_matrix_power(
        rotation_matrix(N_QUBITS, state),
        SUCC_ANGLE,
    )
    for state in range(2**N_QUBITS)
]

FAIL_ROTS = [
    fractional_matrix_power(
        rotation_matrix(N_QUBITS, state),
        FAIL_ANGLE,
    )
    for state in range(2**N_QUBITS)
]

state = np.ones(2**N_QUBITS, dtype=complex) / np.sqrt(2**N_QUBITS)
for t in range(HORIZON):
    arm = sample(state, n_samples=1)[0]
    succ = np.random.rand() < P_LIST[arm]
    rot = SUCC_ROTS[arm] if succ else FAIL_ROTS[arm]
    state = rot @ state

    regrets[t] = np.max(P_LIST) - P_LIST[arm]
    arms[t] = arm
    succs[t] = succ
    # print(f"t={t}, arm={arm}, succ={succ}, regret={regrets[t]}")
    print(state.real)

df = pd.DataFrame({"arm": arms, "succ": succs, "regret": regrets})

# %%
plt.plot(df["regret"].cumsum(), label="regret")
plt.plot(
    np.arange(HORIZON) * (np.max(P_LIST) - np.mean(P_LIST)), label="random"
)
plt.legend()
plt.show()


plt.title("Cum mean success")
plt.plot(df["succ"].cumsum() / np.arange(1, HORIZON + 1))
plt.axhline(np.mean(P_LIST), color="black", linestyle="--")
plt.axhline(np.max(P_LIST), color="black", linestyle="--")
# plt.xscale("log")
plt.show()
