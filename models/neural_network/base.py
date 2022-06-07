from abc import ABC, abstractmethod
from pandas import DataFrame
from .mlp import MLP


class Base(ABC):

    @abstractmethod
    def train(self, data: DataFrame) -> None:
        pass

    @abstractmethod
    def test(self, network: MLP, data: DataFrame) -> None:
        pass
