from pinn.Pinn import PINN
from torch import nn
from pinn.loss.Weights import Weights


class PinnConfig:
    def __init__(self):
        self.weights = Weights(residual=2.0, initial=1.0,
                               boundary=1.0).normalized()

    def getNeuralNetwork(self) -> PINN:
        return PINN(layers=4, neuronsPerLayer=120, act=nn.Tanh())
