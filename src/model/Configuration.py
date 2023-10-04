from InitialCondition import InitialCondition
from Pinn import PINN
from loss.DiffusionMap import DiffusionMap
from loss.Treatment import Treatment
from simulationSpace.TimespaceDomain import TimespaceDomain
from torch import nn


class Configuration:
    def getDiffusionMap(self, device):
        return DiffusionMap(self.getTimespaceDomain(), device)

    def getTimespaceDomain(self):
        return TimespaceDomain(
            spaceDomains=[(0.0, 100.0), (0.0, 100.0)],
            timeDomain=(0.0, 100.0),
        )

    def getTreatment(self):
        return Treatment(
            absorptionRate=2.0,
            decayRate=0.02,
            dose=0.05,
            firstDoseTime=50.0,
            dosesNum=2,
            timeBetweenDoses=20.0,
        )

    def getInitialCondition(self):
        return InitialCondition((60.0, 60.0), 0.4, 10)

    def getNeuralNetwork(self):
        return PINN(layers=4, neuronsPerLayer=120, act=nn.Tanh())
