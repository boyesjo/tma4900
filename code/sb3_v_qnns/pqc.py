import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
import pandas as pd
import multiprocessing as mp
from pathlib import Path
from env import BernoulliBanditsEnv


EPS = np.finfo(np.float32).eps.item()


def theta_shape(n_qubits, entangle_strat):
    if entangle_strat == "one_to_one":
        return n_qubits - 1
    elif entangle_strat == "circular":
        return n_qubits
    elif entangle_strat == "all_to_all":
        return n_qubits * (n_qubits - 1) // 2
    else:
        raise ValueError(f"Unknown entangle_strat: {entangle_strat}")


def r_zz(theta, wires):
    assert len(wires) == 2
    qml.CNOT(wires=wires)
    qml.RZ(theta, wires=wires[1])
    qml.CNOT(wires=wires)


def cz_wrap(_, wires):
    assert len(wires) == 2
    qml.CZ(wires=wires)


def entangle(
    n_qubits,
    theta,
    learnable=False,
    entangle_strat="circular",
):
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
    n_qubits,
    phi,
    theta=None,
    learnable=False,
    entangle_strat="circular",
):
    assert len(phi) == 2 * n_qubits
    for i in range(n_qubits):
        qml.RZ(phi[i], wires=i)
        qml.RY(phi[n_qubits + i], wires=i)

    entangle(n_qubits, theta, learnable, entangle_strat)


def u_enc(n_qubits, s, lam):
    assert len(s) == n_qubits, f"{len(s)} != {n_qubits}, {s}"
    assert len(lam) == 2 * n_qubits

    for i in range(n_qubits):
        qml.RY(lam[i] * s[i], wires=i)
        qml.RZ(lam[n_qubits + i] * s[i], wires=i)


def get_unlearnable_qnn(
    n_qubits,
    n_layers,
    observables,
    entangle_strat="circular",
    device="default.qubit",
):
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
    n_qubits,
    n_layers,
    observables,
    entangle_strat="circular",
    device="default.qubit",
):
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
    n_qubits,
    n_layers,
    observables,
    entangle_strat="circular",
    learnable=False,
    device="default.qubit",
):
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


class RawPQC(nn.Module):
    def __init__(
        self,
        observables,
        n_layers=1,
        n_state=2,
        entangle_strat="circular",
        learnable=False,
        device="default.qubit",
        post_obs=lambda x: x,
    ):
        super().__init__()
        self.n_actions = len(observables)
        self.n_layers = n_layers
        self.n_qubits = n_state

        self.post_obs = post_obs

        _qnn, shapes = get_qnn(
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
        }

        if learnable:
            init_method["theta"] = lambda x: nn.init.constant_(x, torch.pi)

        self.qnn = qml.qnn.TorchLayer(
            _qnn,
            weight_shapes=shapes,
            init_method=init_method,
        )

    def forward(self, x):
        x = self._obs(x)
        return x

    def get_action(self, x):
        return torch.multinomial(self(x), 1).item()

    def _obs(self, x):
        return self.post_obs(self.qnn(x))


class Policy(nn.Module):
    def __init__(self, n_states=4, n_actions=2):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        x = self.layers(x)
        return F.softmax(x, dim=0)


class SoftmaxPQC(RawPQC):
    def __init__(
        self,
        n_states=5,
        n_actions=2,
    ):
        assert n_actions == 2
        n_layers = 1
        super().__init__(
            observables=[
                lambda: qml.expval(
                    qml.PauliZ(0)
                    @ qml.PauliZ(1)
                    @ qml.PauliZ(2)
                    @ qml.PauliZ(3)
                ),
                # lambda: qml.expval(
                #     qml.PauliX(0)
                #     @ qml.PauliX(1)
                #     @ qml.PauliX(2)
                #     @ qml.PauliX(3)
                # ),
            ],
            n_layers=n_layers,
            n_state=n_states,
            entangle_strat="all_to_all",
            learnable=False,
            device="default.qubit",
            post_obs=lambda x: torch.cat([x, -x], dim=-1),
        )
        self.beta = 1.0
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = self._obs(x)
        x = self.w * x
        return torch.softmax(self.beta * x, dim=-1)


class Reinforce:
    def __init__(
        self,
        env,
        policy,
    ):
        self.env = env
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.policy = policy(self.n_states, self.n_actions)
        self.optimizer = optim.Adam(
            [
                {"params": self.policy.qnn.phi, "lr": 0.01},
                {"params": self.policy.qnn.lam, "lr": 0.1},
                {"params": self.policy.w, "lr": 0.1},
            ],
            lr=0.01,
        )
        self.gamma = 1.0

        self.state_normalizer = torch.ones(self.n_states)

        self.actions_history = []
        self.states_history = []
        self.returns_history = []

        self.mean_episode_lengths = []

        self.timesteps = 0
        self.max_timesteps = 250_000

        self.all_actions = []

    def select_action(self, state):
        probs = self.policy(state)
        action = torch.multinomial(probs, 1)
        return action

    def get_returns(self, rewards):
        returns = torch.zeros(len(rewards))
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        returns = (returns - returns.mean()) / (returns.std() + EPS)
        return returns

    def loss(
        self,
        states,
        actions,
        returns,
        batch_size=1,
    ):
        log_probs = torch.log(self.policy(states))
        log_probs_taken = log_probs[torch.arange(len(states)), actions]
        loss = -(log_probs_taken * returns.detach()).sum() / batch_size
        return loss

    def play_episode(self):
        states = []
        rewards = []
        actions = []

        state = self.env.reset()
        state = torch.tensor(state) / self.state_normalizer
        done = False
        while not done:
            self.timesteps += 1

            states.append(state)
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            state = torch.tensor(state) / self.state_normalizer

            rewards.append(reward)
            actions.append(action)

            self.all_actions.append(action.item())

            if self.timesteps > self.max_timesteps:
                break

        return (
            torch.stack(states),
            torch.concat(actions),
            torch.tensor(rewards),
        )

    def train_once(self):
        with torch.no_grad():
            states, actions, rewards = self.play_episode()
            returns = self.get_returns(rewards)

            self.states_history.append(states)
            self.actions_history.append(actions)
            self.returns_history.append(returns)

            episode_length = len(rewards)
            self.mean_episode_lengths.append(episode_length)

    def train_batch(self, batch_size: int = 10):
        self.optimizer.zero_grad()

        self.states_history = []
        self.actions_history = []
        self.returns_history = []

        for _ in range(batch_size):
            if self.timesteps > self.max_timesteps:
                break
            self.train_once()

        states = torch.cat(self.states_history)
        actions = torch.cat(self.actions_history)
        returns = torch.cat(self.returns_history)

        loss = self.loss(states, actions, returns, batch_size)
        loss.backward()

        self.optimizer.step()


def train_and_eval(
    env,
    policy,
    num_timesteps=250_000,
):
    agent = Reinforce(env, policy)

    agent.max_timesteps = num_timesteps
    while agent.timesteps < agent.max_timesteps:
        agent.train_batch(10)

    actions = np.array(agent.all_actions)
    return actions


def train_and_save(
    env,
    policy,
    filename,
    num_timesteps=250_000,
):
    logger.info(f"Training {filename}")
    actions = train_and_eval(env, policy, num_timesteps)
    df = pd.DataFrame(actions)
    df.to_csv(f"{filename}.csv", index=False, header=False)
    logger.success(f"Saved {filename}")


def prior(arms):
    return np.linspace(0.5, 0.505, arms)


if __name__ == "__main__":
    SAVE_DIR = Path("saved_models")

    # total_timesteps = 250_000
    total_timesteps = 1000

    # n_sims = 10
    n_sims = 2

    env = BernoulliBanditsEnv(
        min_turns=10,
        max_turns=1000,
        arms=2,
        prior=prior,
    )

    with mp.Pool() as pool:
        pool.starmap(
            train_and_save,
            [
                (env, SoftmaxPQC, SAVE_DIR / f"qnn_{i}", total_timesteps)
                for i in range(n_sims)
            ],
        )
