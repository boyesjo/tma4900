import numpy as np
from scipy.stats import beta


def thompson(
    p_list: np.ndarray,
    horizon: int,
) -> np.ndarray:
    posteriors = [{"a": 1, "b": 1} for _ in p_list]
    regret = np.zeros(horizon)

    p_max = np.max(p_list)

    for t in range(horizon):
        arm = np.argmax([beta.rvs(**post) for post in posteriors])
        reward = np.random.binomial(1, p_list[arm])
        posteriors[arm]["a"] += reward
        posteriors[arm]["b"] += 1 - reward
        regret[t] = p_max - p_list[arm]

    return np.cumsum(regret)
