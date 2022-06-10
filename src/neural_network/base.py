from abc import ABC, abstractmethod


class Base(ABC):

    @abstractmethod
    def train(self, data: tuple) -> None:
        pass

    @abstractmethod
    def test(self, data: tuple) -> tuple:
        pass
