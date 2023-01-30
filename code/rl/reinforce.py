# %%
from typing import Sequence

import gym
import numpy as np
import pandas as pd
import pennylane as qml
import pqcs
import torch
from torch import nn, optim

# %%
ENV_NAME = "CartPole-v1"
BATCH_SIZE = 10
EPOCHS = 2000 // BATCH_SIZE
GAMMA = 1.0
N_LAYERS = 1
STATE_NORMALISER = np.array(
    [
        2.5,
        2.5,
        0.25,
        2.5,
    ]
)

model = pqcs.SoftmaxPQC(
    n_layers=N_LAYERS,
    n_state=4,
    init_w=torch.ones(1),
    entangle_strat="all_to_all",
    learnable=False,
    observables=[
        lambda: qml.expval(
            qml.PauliX(0)  # @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3)
        ),
    ],
    post_obs=lambda x: torch.cat([x, -x], dim=1),
)

# print number of parameters
print(sum(p.numel() for p in model.parameters()))

# model = pqcs.RawPQC(
#     n_layers=N_LAYERS,
#     n_state=4,
#     entangle_strat="all_to_all",
#     observables=[
#         lambda: qml.expval(
#             qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2) @ qml.PauliX(3)
#         ),
#     ],
#     post_obs=lambda x: (torch.cat([x, -x], dim=1) + 1) / 2,
# )


# %%
def get_trajectories(
    batch_size: int,
    env_name: str,
    model: nn.Module,
) -> list[dict[str, list[float]]]:

    env = gym.make(env_name)
    trajectories: list[dict[str, list[float]]] = [
        {
            "states": [],
            "actions": [],
            "rewards": [],
        }
        for _ in range(batch_size)
    ]

    for i in range(batch_size):
        state = env.reset()
        done = False
        while not done:
            state /= STATE_NORMALISER
            trajectories[i]["states"].append(state)

            action = model.get_action(
                torch.tensor(np.array([state])),
            )  # type: ignore
            state, reward, done, _ = env.step(action)

            trajectories[i]["actions"].append(action)
            trajectories[i]["rewards"].append(reward)

    return trajectories


def get_returns(
    rewards: Sequence[float],
    gamma: float = 1.0,
) -> list[float]:

    returns = [0.0] * len(rewards)
    for i in range(len(rewards) - 2, -1, -1):
        returns[i] = rewards[i] + gamma * returns[i + 1]

    # standardise returns
    arr = np.array(returns)
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
    returns = arr.tolist()

    return returns


# %%
rewards = np.zeros((EPOCHS, BATCH_SIZE))

optimiser = optim.Adam(
    [
        {"params": model.qnn.phi, "lr": 0.01},
        {"params": model.qnn.lam, "lr": 0.1},
        {"params": model.w, "lr": 0.1},
    ],
    lr=0,
)

for epoch in range(EPOCHS):

    with torch.no_grad():
        trajectories = get_trajectories(BATCH_SIZE, ENV_NAME, model)

    for trajectory in trajectories:
        trajectory["returns"] = get_returns(trajectory["rewards"], GAMMA)

    states = np.concatenate(
        [trajectory["states"] for trajectory in trajectories]
    )
    actions = np.concatenate(
        [trajectory["actions"] for trajectory in trajectories]
    ).astype(int)
    returns = np.concatenate(
        [trajectory["returns"] for trajectory in trajectories]
    )

    loss = (
        -torch.sum(
            torch.tensor(returns).detach()
            * torch.log(
                model(torch.tensor(states))[np.arange(len(states)), actions]
            )
        )
        / BATCH_SIZE
    )

    loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    # print(loss.item(), model.w.detatch().numpy())

    rewards[epoch] = np.array(
        [sum(trajectory["rewards"]) for trajectory in trajectories]
    )

    print(f"Epoch {epoch}: {sorted(rewards[epoch])}")


torch.save(model.state_dict(), "model.pt")

pd.DataFrame(rewards).to_csv(
    "rewards.csv",
    index_label="epoch",
    columns=np.arange(BATCH_SIZE),
)

# %%
