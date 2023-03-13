import numpy as np


def ucb(
    p_list: np.ndarray,
    horizon: int,
) -> np.ndarray:

    arm_counts = np.zeros_like(p_list)
    arm_sums = np.zeros_like(p_list)
    regret = np.zeros(horizon)

    p_max = np.max(p_list)

    for arm in range(len(p_list)):
        arm_sums[arm] = np.random.binomial(1, p_list[arm])
        arm_counts[arm] += 1
        regret[arm] = p_max - p_list[arm]

    for t in range(len(p_list), horizon):
        arm = np.argmax(
            arm_sums / arm_counts + np.sqrt(2 * np.log(t) / arm_counts)
        )  # type: int
        regret[t] = p_max - p_list[arm]

        reward = np.random.binomial(1, p_list[arm])
        arm_sums[arm] += reward
        arm_counts[arm] += 1

    return np.cumsum(regret)
