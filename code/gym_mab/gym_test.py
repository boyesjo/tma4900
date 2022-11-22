# %%
from typing import Optional

import gym
import numpy as np
from gym import spaces


class BanditsEnv(gym.Env):

    metadata = {"render.modes": ["print", "pyplot"]}

    def __init__(self, arms: int, max_turns: int = 100):
        self.amrs = arms
        self.max_turns = max_turns

        self.action_space = spaces.Discrete(arms)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(
                3,
                arms,
            ),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None) -> np.ndarray:
        self.arms_pulled: list[int] = []
        self.arm_counts = np.zeros(self.amrs)
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
        reward_array = np.zeros(self.amrs)
        reward_array[action] = self.reward_list[-1] if self.reward_list else 0

        return np.array([arm_means, self.arm_counts, reward_array])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        self.turn += 1
        self.arms_pulled.append(action)
        self.arm_counts[action] += 1

        self.reward_list.append(np.random.binomial(1, self.p_list[action]))
        reward = sum(self.reward_list)

        done = self.turn >= self.max_turns

        obs = self._get_obs(action)
        return obs, reward, done, {}

    def render(
        self, mode: str = "print", action: Optional[int] = None, **kwargs
    ) -> None:
        if mode == "print":
            print(
                # self.p_list,
                action if action is not None else "",
                np.argmax(self.p_list),
                # self.reward_list[-1],
                sep="\t",
            )

        if mode == "pyplot":
            import matplotlib.pyplot as plt

            regret = self.regret
            # plt.title(f"Arm means: {self.p_list}")
            plt.plot(regret)
            plt.show()

    @property
    def regret(self) -> float:
        return np.cumsum(np.max(self.p_list) - self.p_list[self.arms_pulled])


# %%
if __name__ == "__main__":
    import stable_baselines3 as sb3

    N = 20

    agents = {
        "A2C": sb3.A2C,
        "PPO": sb3.PPO,
        "DQN": sb3.DQN,
    }

    sb3.DQN().predict()

    env = BanditsEnv(arms=3, max_turns=100)

    regrets = {name: np.zeros((N, env.max_turns)) for name in agents.keys()}

    for name, agent in agents.items():

        model = agent("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=1e6)

        obs = env.reset()
        for i in range(N * env.max_turns):
            action, _state = model.predict(obs)  # , deterministic=True)
            obs, reward, done, info = env.step(action)
            # env.render(action=action)
            if done:
                j = i // env.max_turns
                regrets[name][j] = env.regret
                obs = env.reset()

    import matplotlib.pyplot as plt

    for name, regret in regrets.items():
        plt.plot(np.mean(regret, axis=0), label=name)
        # add standard deviation
        plt.fill_between(
            np.arange(env.max_turns),
            np.mean(regret, axis=0) - np.std(regret, axis=0) / (N + 1),
            np.mean(regret, axis=0) + np.std(regret, axis=0) / (N + 1),
            alpha=0.2,
        )

    plt.legend()
    plt.show()

# %%
