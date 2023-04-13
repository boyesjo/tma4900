# %%
import numpy as np
from scipy.stats import norm, beta
import matplotlib.pyplot as plt


# %%
def gen_p_list(a, b, sigma):
    p1 = beta.rvs(a, b)
    logit_p1 = np.log(p1 / (1 - p1))
    logit_p2 = logit_p1 + norm.rvs(scale=sigma)
    p2 = 1 / (1 + np.exp(-logit_p2))
    return np.array([p1, p2])


N_SAMPLES = int(1e6)

p_list = np.array([gen_p_list(2, 2, 0.02) for _ in range(N_SAMPLES)])

# %%
# get mean absolute difference
np.mean(np.abs(p_list[:, 0] - p_list[:, 1]))


# %%
# plot hist of abs diff
plt.hist(np.abs(p_list[:, 0] - p_list[:, 1]), bins=100)
# %%
# a = 0.5, b = 0.5, sigma = 0.1
# 0.009964566867240768

# a = 2, b = 2, sigma = 0.02
# 0.00319093834230492
