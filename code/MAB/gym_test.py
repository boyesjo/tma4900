from typing import Optional

import gym
import numpy as np
from gym import spaces


class BanditsEnv(gym.Env):

    metadata = {"render.modes": ["print"]}

    def __init__(self, arms: int, max_turns: int = 100):
        self.amrs = arms
        self.max_turns = max_turns

        self.action_space = spaces.Discrete(arms)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                2,
                arms,
            ),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> np.ndarray:
        self.arms_pulled: list[int] = []
        self.p_list = np.random.uniform(0, 1, self.amrs)
        self.turn = 0
        self.reward_list: list[float] = []
        return self._get_obs(0)

    def _get_obs(self, action: int) -> np.ndarray:
        arm_means = np.array(
            [
                np.nan_to_num(
                    np.mean(
                        [
                            self.reward_list[i]
                            for i in range(len(self.reward_list))
                            if self.arms_pulled[i] == arm
                        ]
                    )
                )
                for arm in range(self.amrs)
            ]
        )
        reward = np.zeros(self.amrs)
        reward[action] = self.reward_list[-1] if self.reward_list else 0

        return np.array([arm_means, reward])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.turn += 1
        self.arms_pulled.append(action)

        self.reward_list.append(np.random.binomial(1, self.p_list[action]))
        reward = sum(self.reward_list) / self.turn
        reward = self.reward_list[-1]

        done = self.turn >= self.max_turns

        obs = self._get_obs(action)
        return obs, reward, done, {}

    def render(
        self, mode: str = "print", action: Optional[int] = None
    ) -> None:
        if mode == "print":
            print(
                self.p_list,
                action if action is not None else "",
                np.argmax(self.p_list),
                self.reward_list[-1],
                sep="\t",
            )


if __name__ == "__main__":
    from stable_baselines3 import A2C

    env = BanditsEnv(arms=2, max_turns=200)

    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20_000)

    obs = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render(action=action)
        if done:
            obs = env.reset()
