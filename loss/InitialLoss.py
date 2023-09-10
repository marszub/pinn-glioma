from typing import Callable
from simulationSpace.SampleSpace import SampleSpace
import torch
from Pinn import PINN

class InitialLoss:
    def __init__(
        self,
        space: SampleSpace,
        initialCondition: Callable,
    ):
        self.space = space
        self.initialCondition = initialCondition

    def __initial(self, pinn: PINN):
        x, y, t = self.space.getInitialPoints()
        pinn_init = self.initialCondition(x, y)
        loss = pinn(x, y, t) - pinn_init
        return loss.pow(2).mean()

    def verbose(self, pinn: PINN):
        self.space.to(pinn.device())

        initial_loss = self.__initial(pinn)

        return initial_loss, torch.zeros_like(initial_loss), initial_loss, torch.zeros_like(initial_loss)

    def __call__(self, pinn: PINN):
        return self.verbose(pinn)[0]