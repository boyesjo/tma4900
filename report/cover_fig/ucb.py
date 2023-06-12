# %%
import numpy as np
import pandas as pd

P_LIST = [
    0.69,
    0.1337,
    0.42,
    0.314,
    0.666,
    0.3,
    0.46,
    0.63,
]

N_TURNS = 1000
N_TURNS_Q = 10000


def upper(turn, count):
    return np.sqrt(2 * np.log(turn) / count)


def ucb(p_list, n_turns):
    rewards = np.zeros_like(p_list)
    counts = np.zeros_like(p_list)
    for turn in range(1, n_turns + 1):
        if turn <= len(p_list):
            arm = turn - 1
        else:
            arm = np.argmax(rewards / counts + upper(turn, counts))
        reward = np.random.binomial(1, p_list[arm])
        rewards[arm] += reward
        counts[arm] += 1

    print(rewards, counts)
    return rewards / counts, upper(n_turns, counts)


# %%
est, up = ucb(P_LIST, N_TURNS)
est_q, up_q = ucb(P_LIST, N_TURNS_Q)

pd.DataFrame(
    {
        "arm": np.arange(1, len(P_LIST) + 1),
        "est": est,
        "up": up,
        "est_q": est_q,
        "up_q": up_q,
    }
).to_csv("ucb.dat", index=False)

pd.DataFrame(
    {
        "arm": np.arange(1, len(P_LIST) + 1),
        "p": P_LIST,
    }
).to_csv("p.dat", index=False)

# %%
