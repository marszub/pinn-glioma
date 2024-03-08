from model.loss.InitialCondition import InitialCondition
from model.Pinn import PINN
from model.loss.LoadedDiffusionMap import LoadedDiffusionMap
from model.loss.Treatment import Treatment
from model.simulationSpace.TimespaceDomain import TimespaceDomain
from torch import nn


class Configuration:
    def getDiffusionMap(self, device):
        return LoadedDiffusionMap(self.getTimespaceDomain(), device, "resources/diffusion_map.npy")

    def getTimespaceDomain(self):
        return TimespaceDomain(
            spaceDomains=[(0.0, 50.0), (0.0, 80.0)],
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
        return InitialCondition((28.0, 45.0), 0.4, 5)

    def getNeuralNetwork(self):
        return PINN(layers=4, neuronsPerLayer=120, act=nn.Tanh())
