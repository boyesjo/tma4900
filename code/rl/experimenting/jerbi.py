# %%
import pennylane as qml

# from pennylane import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
from typing import Sequence
from collections import deque, defaultdict


# %%
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


get_qnn(n_qubits=2, n_layers=2)(
    inputs=[0.1, 0.2],
    phi=[
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ],
    lam=[
        [0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.3, 0.4],
    ],
)


# %%
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
def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name):

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    states = [e.reset() for e in envs]

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = np.array(
            [s / state_bounds for i, s in enumerate(states) if not done[i]]
        )

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]["states"].append(state)

        # convert to torch tensor
        states = torch.tensor(normalized_states, dtype=torch.float32)
        action_probs = model(states)

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs):
            action = np.random.choice(n_actions, p=policy.detach().numpy())
            states[i], reward, done[i], _ = envs[i].step(action)
            trajectories[i]["actions"].append(action)
            trajectories[i]["rewards"].append(reward)

    return trajectories


def compute_returns(rewards_history, gamma):
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()
    return returns


env_name = "CartPole-v1"
env = gym.make(env_name)
n_actions = env.action_space.n
n_obs = env.observation_space.shape[0]

model = QDQN(n_obs=n_obs, n_actions=n_actions, n_layers=5)

state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
gamma = 1
batch_size = 10
n_episodes = 1000


def reinforce_update(states, actions, returns, model):

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    returns = torch.tensor(returns)

    p_actions = model(states)
    # select columns of p_actions corresponding to actions
    p_actions = p_actions[range(len(p_actions)), actions]

    log_probs = torch.log(p_actions)

    d_theta = torch.sum(log_probs * returns.detach()) / batch_size
    d_theta.backward()

    with torch.no_grad():
        # lr of 0.1 for lam
        model.qnn.lam += 0.1 * model.qnn.lam.grad
        model.qnn.lam.grad = None

        # lr of 0.01 for phi
        model.qnn.phi += 0.01 * model.qnn.phi.grad
        model.qnn.phi.grad = None


episode_rewards = []

for batch in range(n_episodes // batch_size):
    with torch.no_grad():
        episodes = gather_episodes(
            state_bounds, n_actions, model, batch_size, env_name
        )

    states = np.concatenate([e["states"] for e in episodes])
    actions = np.concatenate([e["actions"] for e in episodes])
    rewards = [ep["rewards"] for ep in episodes]
    returns = np.concatenate(
        [compute_returns(ep_rwds, gamma) for ep_rwds in rewards]
    )

    # Update model parameters.
    reinforce_update(states, actions, returns, model)

    # Store collected rewards
    for ep_rwds in rewards:
        episode_rewards.append(np.sum(ep_rwds))

    avg_rewards = np.mean(episode_rewards[-batch_size:])

    print(
        "Finished episode",
        (batch + 1) * batch_size,
        "Average rewards: ",
        avg_rewards,
    )

    if avg_rewards >= 500.0:
        break

# %%
# save model
torch.save(model.state_dict(), "model.pt")

# %%
plt.plot(episode_rewards)
plt.savefig("reinforce_cartpole.png")

import pandas as pd

pd.DataFrame(episode_rewards).to_csv(
    "reinforce_cartpole.csv", index_label="episode", header=["reward"]
)
# %%

pd.DataFrame(episode_rewards).rolling(50).mean().plot()
plt.savefig("reinforce_cartpole_rolling50.png")

# %%
