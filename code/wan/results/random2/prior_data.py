# %%
import pandas as pd
import numpy as np
from scipy.stats import beta, norm

# %%
a = 0.5
b = 0.5

sigma = 0.1

RESOLUTION = 100

# %%
mu_1_list = np.linspace(0, 1, RESOLUTION)
mu_2_list = np.linspace(0, 1, RESOLUTION)
p_arr = np.zeros((RESOLUTION, RESOLUTION))

for i, mu_1 in enumerate(mu_1_list):
    p_one = beta.pdf(mu_1, a, b)
    for j, mu_2 in enumerate(mu_2_list):
        logit_mu1 = np.log(mu_1 / (1 - mu_1))
        logit_mu2 = np.log(mu_2 / (1 - mu_2))
        p_two = norm.pdf(logit_mu2 - logit_mu1, scale=sigma)
        p_arr[i, j] = p_one * p_two * norm.pdf(mu_2 - mu_1, scale=sigma)


# %%
import matplotlib.pyplot as plt

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
X, Y = np.meshgrid(mu_1_list, mu_2_list)
ax.plot_surface(X, Y, p_arr, cmap="viridis")
plt.show()
# %%
