# %%
import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback

ENV_NAME = "CartPole-v1"
EPOCHS = 2000
MAX_STEPS = 500


class Callback(BaseCallback):
    def __init__(self, verbose=0):
        super(Callback, self).__init__(verbose)
        self.rewards = [[]]
        self.epochs = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        assert len(dones) == 1
        if dones[0]:
            self.epochs += 1
            self.rewards.append([])
        rewards = self.locals.get("rewards")
        assert len(rewards) == 1
        self.rewards[-1].append(rewards[0])

        return self.epochs < EPOCHS


model = sb3.A2C(
    "MlpPolicy",
    ENV_NAME,
    # verbose=1,
)
callback = Callback()


# train the agent and plot mean reward for each epoch
model.learn(
    total_timesteps=EPOCHS * MAX_STEPS,
    callback=callback,
)


sums = np.array([sum(x) for x in callback.rewards])
plt.plot(sums)
# %%
