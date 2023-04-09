# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from loguru import logger

EPS = np.finfo(np.float32).eps.item()


# Define the policy network
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.fc2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.softmax(x, dim=0)


class Reinforce:
    def __init__(
        self,
        env,
        policy,
    ):
        self.env = env
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=1e-3)
        self.gamma = 1.0

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
        self, log_probs: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        return -log_probs * returns

    def play_episode(self):
        states = []
        rewards = []
        log_probs = []

        state = self.env.reset()
        state = torch.tensor(state)
        done = False
        while not done:

            states.append(state)
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            state = torch.tensor(state)

            rewards.append(reward)
            log_prob = torch.log(self.policy(state))[action] + EPS
            log_probs.append(log_prob)

        self.states = torch.concat(states)
        self.rewards = torch.tensor(rewards)
        self.log_probs = torch.concat(log_probs)

    def train_once(self):
        self.optimizer.zero_grad()
        self.play_episode()
        returns = self.get_returns(self.rewards)
        loss = self.loss(self.log_probs, returns)
        loss.sum().backward()
        logger.success(f"loss: {loss.sum()}, len: {len(returns)}")
        self.optimizer.step()

        # print average parameter norm
        total_norm = 0
        for p in self.policy.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        logger.info(f"grad norm: {total_norm}")


# %%
env = gym.make("CartPole-v1")
policy = Policy()
reinforce = Reinforce(env, policy)

for i in range(10000):
    reinforce.train_once()

# %%
