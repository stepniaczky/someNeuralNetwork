from dataclasses import dataclass
from random import random


@dataclass
class NeuralLayer:
    nr_inputs: int
    nr_outputs: int
    inputs: list
    use_bias: bool
    activation_f: int
    derivative_or_activation_f: int
    neurons: list[list[float]]
    gradients: list[list[int]]

    def __post_init__(self):
        self.inputs += 1
        self.neurons = [[random() for _ in range(self.nr_inputs)] for _ in range(self.nr_outputs)]
        self.gradients = [[0 for _ in range(self.nr_inputs)] for _ in range(self.nr_outputs)]
