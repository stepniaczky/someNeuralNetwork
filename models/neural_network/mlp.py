from .base import Base, NeuralNetwork
from pandas import DataFrame


class MLP(Base):

    def __init__(self, _dict: dict):
        self.initial = _dict

    def train(self, data: DataFrame) -> None:
        ...

    def test(self, network: NeuralNetwork, data: DataFrame) -> None:
        ...
