# %%
import matplotlib.pyplot as plt
import numpy as np
from oracle import Oracle
from qmc import qmc

np.seterr(all="ignore")

# %%
idk = 0


def qucb1(p_list: np.ndarray, horizon: float, delta: float = 0.1):
    C1 = 2
    n_arms = len(p_list)

    def n_turns(r: float) -> int:
        return int(np.ceil(C1 * np.log(1 / delta) / r))

    r_list = np.ones_like(p_list)
    oracle_list = [Oracle(p, 1) for p in p_list]
    estimate_list = np.zeros_like(p_list)

    arms_played = []
    times_played = []

    for arm in range(n_arms):
        oracle = oracle_list[arm]
        n = n_turns(r_list[arm])

        estimate, n_queries = qmc(oracle, n, delta, method="canonical")
        estimate_list[arm] = estimate
        arms_played.append(arm)
        times_played.append(n_queries)

    global idk
    idk = sum(times_played)

    while sum(times_played) < int(horizon):

        arm = np.argmax(estimate_list + r_list)
        oracle = oracle_list[arm]
        r_list[arm] /= 2
        n = n_turns(r_list[arm])

        estimate, n_queries = qmc(oracle, n, delta, method="canonical")
        for val in (
            arm,
            estimate_list[arm],
            estimate,
            r_list[arm],
            p_list[arm],
        ):
            # print with 2 decimal places if float
            if isinstance(val, float):
                print(f"{val:.2f}", end=" ")
            else:
                print(val, end=" ")
        print()

        estimate_list[arm] = (estimate + estimate_list[arm]) / 2
        arms_played.append(arm)
        times_played.append(n_queries)

    return arms_played, times_played


def regret(p_list, arms_played, times_played):
    """Given a list of probabilities, a list of each arm played in order
    and a list of the number of times the respecive arm was played at that time
    return the regret for each time step.
    """
    ret = []
    p_max = max(p_list)
    for arm, n in zip(arms_played, times_played):
        p = p_list[arm]
        ret += [p_max - p] * n
    return ret


# %%
def ucb(p_list: np.ndarray, horizon: float, delta: float = 0.1):
    # regular ucb1 alg for comparison
    n_arms = len(p_list)
    est_list = np.zeros_like(p_list)
    times_pulled = np.zeros_like(p_list)
    reg = []

    for arm in range(n_arms):
        est_list[arm] = np.random.binomial(1, p_list[arm])
        times_pulled[arm] += 1
        reg.append(max(p_list) - p_list[arm])

    for t in range(int(horizon) - n_arms):
        arm = np.argmax(est_list + np.sqrt(2 * np.log(t) / times_pulled))
        reg.append(max(p_list) - p_list[arm])

        reward = np.random.binomial(1, p_list[arm])
        est_list[arm] = (est_list[arm] * times_pulled[arm] + reward) / (
            times_pulled[arm] + 1
        )

        times_pulled[arm] += 1

    return reg


T = 5000

p_list = np.random.uniform(0.01, 0.999, 16)
res = qucb1(p_list, T)


# %%
plt.plot(np.cumsum(regret(p_list, *res)), label="qucb1")
plt.plot(np.cumsum(ucb(p_list, T)), label="ucb1")
plt.xscale("log")
plt.axvline(x=idk, color="red")
plt.legend()
plt.show()

plt.plot(np.cumsum(regret(p_list, *res)), label="qucb1")
plt.plot(np.cumsum(ucb(p_list, T)), label="ucb1")
# plt.xscale("log")
# plt.axvline(x=idk, color="red")
plt.legend()
plt.show()

# %%
