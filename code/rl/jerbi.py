# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from loguru import logger
from pqcs import RawPQC
import pennylane as qml

EPS = np.finfo(np.float32).eps.item()


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
        n_states: int = 4,
        n_actions: int = 2,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        self.state_normalizer = torch.tensor(
            [
                2.4,
                2.4,
                0.2,
                2.5,
            ]
        )

        self.actions_history = []
        self.states_history = []
        self.returns_history = []

        self.use_baseline = False

        self.mean_episode_lengths = []

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        probs = self.policy(state)
        action = torch.multinomial(probs, 1)
        return action

    def get_returns(self, rewards: list[float]) -> torch.Tensor:
        returns = torch.zeros(len(rewards))
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        returns = (returns - returns.mean()) / (returns.std() + EPS)
        return returns

    def loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        batch_size: int = 1,
    ) -> torch.Tensor:
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

            states.append(state)
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            state = torch.tensor(state) / self.state_normalizer

            rewards.append(reward)
            actions.append(action)

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
            logger.debug(f"episode length: {episode_length}")

    def train_batch(self, batch_size: int = 10):
        self.optimizer.zero_grad()

        self.states_history = []
        self.actions_history = []
        self.returns_history = []

        for _ in range(batch_size):
            self.train_once()

        self.fit_baseline()
        states = torch.cat(self.states_history)
        actions = torch.cat(self.actions_history)
        returns = torch.cat(self.returns_history)

        if self.use_baseline:
            returns = returns - self.baseline(states)

        loss = self.loss(states, actions, returns, batch_size)
        loss.backward()

        logger.info(f"epsiode length: {int(len(returns)/batch_size):3}")
        logger.info(f"loss: {loss.item():.3f}")

        self.optimizer.step()

    def _pre_baseline(self, states: torch.Tensor) -> torch.Tensor:
        t = torch.arange(len(states))
        return torch.cat(
            [
                states,
                states**2,
                0.01 * t.unsqueeze(1),
                (0.01 * t.unsqueeze(1)) ** 2,
                (0.01 * t.unsqueeze(1)) ** 3,
                torch.ones(len(states), 1),
            ],
            dim=1,
        )

    def fit_baseline(self) -> None:
        x = torch.cat(self.states_history)
        x = self._pre_baseline(x)
        y = torch.cat(self.returns_history)
        params = torch.linalg.inv(x.T @ x) @ x.T @ y
        self.baseline_params = params

    def baseline(self, new_states: torch.Tensor) -> torch.Tensor:
        x = self._pre_baseline(new_states)
        return x @ self.baseline_params


# %%
def test_reinforce(_) -> np.ndarray:
    env = gym.make("CartPole-v1")
    reinforce = Reinforce(env, SoftmaxPQC)

    for i in range(200):
        logger.info(f"batch: {i}")
        reinforce.train_batch(10)

    return np.asarray(reinforce.mean_episode_lengths)


# %%
if __name__ == "__main__":
    import pandas as pd
    import multiprocessing

    N_SIMS = 10
    with multiprocessing.Pool() as pool:
        results = pool.map(test_reinforce, range(N_SIMS))

    df = pd.DataFrame(results)
    df.to_csv("qreinforce.csv")
