from abc import ABC, abstractmethod

from scipy.stats import rv_continuous, rv_discrete


class Arm(ABC):
    @abstractmethod
    def pull(self) -> float:
        pass

    @property
    @abstractmethod
    def dist(self) -> rv_continuous | rv_discrete:
        pass

    @property
    def mean(self) -> float:
        return self.dist.mean()

    @property
    def min_val(self) -> float:
        return self.dist.support()[0]

    @property
    def max_val(self) -> float:
        return self.dist.support()[1]

    @property
    def is_continuous(self) -> bool:
        return isinstance(self.dist, rv_continuous)
