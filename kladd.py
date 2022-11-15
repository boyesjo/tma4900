from abc import ABC, abstractmethod


class Base(ABC):
    @abstractmethod
    def foo(self, x: int) -> int:
        pass


class Derived(Base):
    def foo(self, x: int, y: float = 0) -> int:
        return x + 1
