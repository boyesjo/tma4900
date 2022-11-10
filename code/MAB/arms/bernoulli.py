from scipy.stats import bernoulli

from .arm import Arm


class BernoulliArm(Arm):
    def __init__(self, p):
        if not 0 <= p <= 1:
            raise ValueError("p must be in [0, 1]")
        self.p = p

    @property
    def dist(self):
        return bernoulli(self.p)

    def pull(self):
        return bernoulli.rvs(self.p)

    def __str__(self):
        return f"BernoulliArm(p={self.p})"
