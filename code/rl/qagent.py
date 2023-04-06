from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Sequence


class QPolicy(BasePolicy):
    def __init__(self, policy_net: nn.Module, **kwargs):
        super(QPolicy, self).__init__(
            policy_net,
            **kwargs,
        )
        self.net = policy_net

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        return self.net.forward(observation)


class QAgent(BaseAlgorithm):
    def __init__(
        self,
        policy_net: nn.Module,
        learning_rate=0.01,
        batch_size=10,
        **kwargs,
    ):

        super(QAgent, self).__init__(
            learning_rate=learning_rate,
            policy=QPolicy(policy_net),
            **kwargs,
        )

        self.net = policy_net
        self.batch_size = batch_size
        self.gamma = 0.99

        self.state_normaliser = np.array(
            [
                2.5,
                2.5,
                0.25,
                2.5,
            ]
        )

        self.learning_rate = learning_rate

    def _setup_model(self) -> None:
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.learning_rate
            if isinstance(self.learning_rate, float)
            else 1e-3,
        )

    def predict(self, observation, deterministic=False):

        observation = torch.tensor(observation, dtype=torch.float32)
        action_probs = self.policy(observation)
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1).numpy()
        else:
            actions = (
                torch.multinomial(action_probs, num_samples=1)
                .squeeze(dim=-1)
                .numpy()
            )
        return actions, torch.log(action_probs)

    def _get_trajectories(
        self,
    ) -> list[dict[str, list[float]]]:

        assert self.env is not None, "No environment set"

        trajectories: list[dict[str, list[float]]] = [
            {
                "states": [],
                "actions": [],
                "rewards": [],
            }
            for _ in range(self.batch_size)
        ]

        for i in range(self.batch_size):
            state = self.env.reset()
            done = False
            while not done:
                state /= self.state_normaliser
                trajectories[i]["states"].append(state)

                action = (
                    self.net.forward(torch.tensor(np.array([state])))
                    .argmax()
                    .item()
                )
                trajectories[i]["actions"].append(action)

                state, reward, done, _ = self.env.step(action)  # type: ignore

                assert isinstance(reward, float)
                trajectories[i]["rewards"].append(reward)

        return trajectories

    def get_returns(
        self,
        rewards: Sequence[float],
    ) -> list[float]:

        returns = [0.0] * len(rewards)
        for i in range(len(rewards) - 2, -1, -1):
            returns[i] = rewards[i] + self.gamma * returns[i + 1]

        # standardise returns
        arr = np.array(returns)
        arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-8)
        returns = arr.tolist()

        return returns

    def learn(self, total_timesteps, callback=None, log_interval=100):

        while self.num_timesteps < total_timesteps:

            with torch.no_grad():
                trajectories = self._get_trajectories()

            for trajectory in trajectories:
                trajectory["returns"] = self.get_returns(trajectory["rewards"])

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
                        self.net(torch.tensor(states))[
                            np.arange(len(states)), actions
                        ]
                    )
                )
                / self.batch_size
            )

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return self


if __name__ == "__main__":

    # test on cartpole
    import gym

    env = gym.make("CartPole-v1")

    net = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 2),
    )

    agent = QAgent(net, env=env)
    agent.learn(total_timesteps=10000)
