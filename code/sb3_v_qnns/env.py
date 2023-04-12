import gym
from gym import spaces
import numpy as np
from typing import Callable, Any

Prior = Callable[[int], np.ndarray]
Info = dict[str, Any]
Observables = np.ndarray


class BernoulliBanditsEnv(gym.Env):

    metadata = {
        "render.modes": ["print", "pyplot"],
    }

    def __init__(
        self,
        *,
        arms: int = 2,
        min_turns: int = 100,
        max_turns: int = 10_000,
        prior: Prior = lambda arms: np.random.random(arms),
    ):
        self.arms = arms
        self.min_turns = min_turns
        self.max_turns = max_turns
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(arms,),  # * 2 + 1,),
            dtype=np.float64,
        )
        self.prior = prior

    def reset(self, seed=None, **kwargs) -> Observables:
        self.times_pulled = np.zeros(self.arms, dtype=int)
        self.results = np.zeros(self.arms, dtype=int)
        self.p_list = self.prior(self.arms)
        self.turn = 0
        observation = self._get_obs()

        return observation

    def _get_obs(self) -> Observables:
        means = self.results / self.times_pulled
        means[np.isnan(means)] = 0
        counts = self.times_pulled / self.times_pulled.sum()
        counts[np.isnan(counts)] = 0

        turn = np.tanh(self.turn)

        return 2 * np.concatenate([means]) - 1  # , counts, [turn]]) - 1

    def _regret_baseline(self) -> float:
        # upper_bound = (
        #     8 * self.arms / (self.p_list.max() - self.p_list.min())
        # ) * np.log(self.turn)
        return (
            1.0 * self.turn * (np.max(self.p_list) - np.min(self.p_list)) / 2
        )

    def _is_done(self) -> bool:
        return self.turn >= self.max_turns
        # if self.turn < self.min_turns:
        #     return False

        # if self.turn >= self.max_turns:
        #     return True

        # return self.regret() >= 1.2 * self._regret_baseline()

    def regret(self) -> float:

        return np.sum((self.p_list.max() - self.p_list) @ self.times_pulled)

    def step(self, action: int) -> tuple[Observables, float, bool, Info]:

        arm = action

        # arm: int | np.int64 = 0
        # if action == 0:
        #     # exploit
        #     arm = np.argmax(self.results / self.times_pulled)
        # else:
        #     # explore
        #     arm = np.random.randint(self.arms)

        p = self.p_list[action]

        result = np.random.binomial(1, p)
        reward = result

        self.times_pulled[arm] += 1
        self.results[arm] += result

        done = self._is_done()
        observation = self._get_obs()
        info = {
            "p_list": self.p_list,
            "times_pulled": self.times_pulled,
            "rewards": self.results,
            "regret": self.regret(),
            "regret_action": self.p_list.max() - self.p_list[arm],
            "turn": self.turn,
        }

        # if done and self.turn < self.max_turns:
        #     reward -= 100
        # elif done and self.turn >= self.max_turns:
        #     reward += 100 * (1 - self.regret() / self._regret_baseline())

        self.turn += 1
        return observation, reward, done, info


def test():
    N_ARMS = 2
    MAX_TURNS = 100

    env = BernoulliBanditsEnv(arms=N_ARMS, min_turns=MAX_TURNS)
    env.reset()
    regret = np.zeros(MAX_TURNS)
    for _ in range(MAX_TURNS):
        obs, reward, done, info = env.step(np.random.randint(N_ARMS))
        # print(obs)
        regret[info["turn"]] = info["regret"]

    # plot final regret
    import matplotlib.pyplot as plt

    # print(regret)
    plt.plot(regret)
    plt.show()


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = BernoulliBanditsEnv()
    print(env.reset())
    print(type(env.reset()))
    check_env(env)

    test()
