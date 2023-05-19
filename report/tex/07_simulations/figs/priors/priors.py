# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, beta


def logit(x):
    return np.log(x / (1 - x))


def expit(x):
    return np.exp(x) / (1 + np.exp(x))


def plot_joint_log_likelihood(
    a,
    b,
    std,
    name,
    y_range=1,
    resolution=5001,
    eps=1e-10,
):
    x1_vals = np.linspace(eps, 1 - eps, resolution)
    dx = x1_vals[1] - x1_vals[0]
    x1_logpdf = beta.logpdf(x1_vals, a=a, b=b)
    x2_vals_logit = logit(x1_vals)
    x2_logpgfs = np.asarray(
        [
            norm.logpdf(x2_vals_logit, loc=mean, scale=std)
            for mean in logit(x1_vals)
        ]
    )

    # shift all x2_pdfs by x1_vals to get x2-x1 on y-axis, pad with zeros
    x2_logpgfs = np.pad(
        x2_logpgfs, ((resolution - 1, 0), (0, 0)), constant_values=-np.inf
    )

    for i in range(resolution):
        x2_logpgfs[:, i] = np.roll(x2_logpgfs[:, i], -i)

    # 2D plot, x1 on x-axis, x2 - x1 on y-axis
    data = x2_logpgfs + x1_logpdf

    # normalise
    data -= np.log(np.sum(np.exp(data) * dx * dx))

    data[data == -np.inf] = np.nan

    # high = np.nanquantile(data, 0.99)
    high = 0
    data[data > high] = high

    low = -100
    data[data < low] = low
    data[np.isnan(data)] = low

    plt.imshow(
        data,
        extent=[0, 1, -1, 1],
        aspect="auto",
        # cmap="virdis",
    )
    plt.ylim(-y_range, y_range)
    plt.axis("off")
    print(f"Saving {name}")
    # plt.savefig(f"{name}.pdf", bbox_inches="tight", pad_inches=0)
    plt.show()


# Define the different global parameters
params = [
    {"a": 0.5, "b": 0.5, "std": 0.1, "name": "prior1", "y_range": 0.5},
    {"a": 2, "b": 2, "std": 0.02, "name": "prior2", "y_range": 0.1},
    # {"a": 1, "b": 2, "std": 10, "name": "prior3"},
]

# Call the function for each set of parameters
for p in params:
    plot_joint_log_likelihood(**p)


# %%
