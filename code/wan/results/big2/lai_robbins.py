# %%
import numpy as np
import pandas as pd


def bernoulli(p_list: np.ndarray, T: int, step: int = 1) -> pd.DataFrame:
    if any(p <= 0 or p >= 1 for p in p_list):
        raise ValueError("p must be in (0, 1)")
    best_idx = np.argmax(p_list)
    best_p = p_list[best_idx]
    kl_divs = np.array(
        [
            p * np.log(p / best_p) + (1 - p) * np.log((1 - p) / (1 - best_p))
            for p in p_list
        ]
    )
    kl_divs = np.delete(kl_divs, best_idx)
    delta_a = np.delete(best_p - p_list, best_idx)
    coeff = sum(delta / kl for delta, kl in zip(delta_a, kl_divs))
    t_list = np.arange(1, T + 1, step=step)
    return pd.DataFrame(
        {"turn": t_list, "regret": coeff * np.log(t_list)},
    ).set_index("turn")


if __name__ == "__main__":
    print(bernoulli(np.array([0.1, 0.9]), 10))

    assert np.allclose(
        bernoulli(np.array([0.1, 0.9]), 10).regret.iloc[1],
        (0.8 * np.log(2))
        / ((0.1 * np.log(0.1 / 0.9) + 0.9 * np.log(0.9 / 0.1))),
    )

# %%
