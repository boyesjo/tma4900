# %%

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
from loguru import logger

EPS = np.finfo(np.float32).eps.item()


class Policy(nn.Module):
    def __init__(self, n_states=4, n_actions=2):
        super(Policy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

        # self.saved_log_probs = []
        # self.rewards = []

    def forward(self, x):
        x = self.layers(x)
        return F.softmax(x, dim=0)


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
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.gamma = 1.0

        self.baseline = torch.zeros(self.n_states + 1)
        self.states_history = []
        self.returns_history = []

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

        # returns = (returns - returns.mean()) / (returns.std() + EPS)
        return returns

    def loss(
        self,
        actions: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        log_probs = torch.log(self.policy(self.states))
        log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        loss = -(log_probs * returns).sum()
        return loss

    def play_episode(self):

        states = []
        rewards = []
        actions = []

        state = self.env.reset()
        state = torch.tensor(state)
        done = False
        while not done:

            states.append(state)
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action.item())

            state = torch.tensor(state)

            rewards.append(reward)
            actions.append(action)

        self.actions = torch.concat(actions)
        self.states = torch.stack(states)
        self.rewards = torch.tensor(rewards)

    def train_once(self):
        self.optimizer.zero_grad()

        with torch.no_grad():
            self.play_episode()
            returns = self.get_returns(self.rewards)

            self.states_history.append(self.states)
            self.returns_history.append(returns)

            self.update_baseline()
            returns -= self.baseline_value(self.states)

        loss = self.loss(self.actions, returns)
        loss.backward()
        logger.info(f"epsiode length: {len(returns)}, loss: {loss}")
        self.optimizer.step()

    def baseline_value(self, state: torch.Tensor) -> torch.Tensor:
        return state @ self.baseline[:-1] + self.baseline[-1]

    def update_baseline(self):

        # cf https://arxiv.org/pdf/1604.06778.pdf

        states = torch.cat(self.states_history)
        t = torch.arange(len(states))

        x = torch.cat(
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

        y = torch.cat(self.returns_history)
        params = torch.linalg.inv(x.T @ x) @ x.T @ y
        self.baseline = params


# %%
env = gym.make("CartPole-v1")
reinforce = Reinforce(env, Policy)

# %%
for i in range(1000):
    reinforce.train_once()


# %%
# play and render the game
import matplotlib.pyplot as plt
from IPython.display import clear_output
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"


env = gym.make("CartPole-v1")
obs = env.reset()
env.seed(0)
score = 0

for _ in range(1000):
    obs, reward, done, info = env.step(env.action_space.sample())
    score += reward
    if done:
        score = 0
        obs = env.reset()
    clear_output(wait=True)
    plt.imshow(env.render(mode="rgb_array"))
    plt.title(f"score: {score}")
    plt.show()

env.close()

# %%
