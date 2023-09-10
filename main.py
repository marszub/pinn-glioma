from loss.DiffusionMap import DiffusionMap
from loss.Treatment import Treatment
from simulationSpace.RandomSpace import RandomSpace
from simulationSpace.TimespaceDomain import TimespaceDomain
from Tracker import Tracker
import torch
from torch import nn
from loss.Loss import Loss
from Pinn import PINN
from InitialCondition import InitialCondition
from Traininer import Trainer
from Weights import Weights
from simulationSpace.UniformSpace import UniformSpace

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timespace = TimespaceDomain(
        spaceDomains=[(0.0, 100.0), (0.0, 100.0)],
        timeDomain=(0.0, 100.0),
    )
    diffusion = DiffusionMap(timespace, device)
    treatment = Treatment(absorptionRate=2.0, decayRate=0.02, dose=0.02, firstDoseTime=10.0, dosesNum=3, timeBetweenDoses=30.0)

    plotSpace = UniformSpace(
        timespaceDomain=timespace,
        spaceResoultion=150,
        timeResoultion=20,
        requiresGrad=False,
    )
    learnRandom = RandomSpace(
        timespaceDomain=timespace,
        initialSize=35*35,
        interiorSize=35*35*35,
        boundarySize=35
    )
    initialCondition = InitialCondition((60.0, 60.0), 0.4, 10)

    tracker = Tracker("tmp", plotSpace)

    pinn = PINN(layers=4, neuronsPerLayer=120, act=nn.Tanh()).to(device)

    weights = Weights(residual=1.0, initial=1.0, boundary=1.0)
    
    loss = Loss(
        learnRandom,
        initialCondition,
        diffusion,
        treatment,
        weights,
    )
    trainer = Trainer(pinn, loss, tracker)
    trainer.train(learning_rate=0.002, max_epochs=100_000)
