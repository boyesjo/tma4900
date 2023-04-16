import stable_baselines3 as sb3
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path
import pandas as pd
import multiprocessing as mp
from env import BernoulliBanditsEnv

SAVE_DIR = Path("saved_models")


class RewardCounter(BaseCallback):
    def __init__(self, verbose=0, total_timesteps=250_000):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.regret = np.zeros(total_timesteps)

    def _on_step(self) -> bool:
        if self.num_timesteps < self.total_timesteps + 1:
            # get regret of action
            self.regret[self.num_timesteps - 1] = self.locals["infos"][-1][
                "regret_action"
            ]
        return True


def train_and_eval(env, agent, num_timesteps=250_000, verbose=0):
    reward_counter = RewardCounter(total_timesteps=num_timesteps)
    model = agent("MlpPolicy", env, verbose=verbose)
    model.learn(total_timesteps=num_timesteps, callback=reward_counter)
    return np.cumsum(reward_counter.regret)


def train_and_save(env, agent, filename, num_timesteps=250_000, verbose=0):
    regret = train_and_eval(env, agent, num_timesteps, verbose)
    df = pd.DataFrame(regret)
    df.to_csv(SAVE_DIR / f"{filename}.csv", index=False, header=False)


def prior(arms):
    return np.linspace(0.5, 0.505, arms)


def main():

    agents = {
        "A2C": sb3.A2C,
        "PPO": sb3.PPO,
        "DQN": sb3.DQN,
    }

    # total_timesteps = 250_000
    total_timesteps = 1000

    # n_sims = 10
    n_sims = 2

    if not SAVE_DIR.exists():
        SAVE_DIR.mkdir()

    env = BernoulliBanditsEnv(
        min_turns=10,
        max_turns=1000,
        arms=2,
        prior=prior,
    )

    for name, agent in agents.items():
        with mp.Pool(mp.cpu_count()) as pool:
            pool.starmap(
                train_and_save,
                [
                    (env, agent, f"{name}_{i}", total_timesteps)
                    for i in range(n_sims)
                ],
            )


if __name__ == "__main__":
    main()
