from Initializer import Initializer
from menu.ArgsParser import ArgsParser
from loss.DiffusionMap import DiffusionMap
from loss.Treatment import Treatment
from simulationSpace.RandomSpace import RandomSpace
from simulationSpace.TimespaceDomain import TimespaceDomain
import torch
from loss.Loss import Loss
from InitialCondition import InitialCondition
from Traininer import Trainer
from Weights import Weights

if __name__ == "__main__":
    argsParser = ArgsParser()
    argsParser.show()
    config = argsParser.get()
    initializer = Initializer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timespace = TimespaceDomain(
        spaceDomains=[(0.0, 100.0), (0.0, 100.0)],
        timeDomain=(0.0, 100.0),
    )
    diffusion = DiffusionMap(timespace, device)
    treatment00 = Treatment(absorptionRate=2.0, decayRate=0.02, dose=0.00, firstDoseTime=10.0, dosesNum=3, timeBetweenDoses=30.0)
    treatment01 = Treatment(absorptionRate=2.0, decayRate=0.02, dose=0.02, firstDoseTime=10.0, dosesNum=3, timeBetweenDoses=30.0)
    treatment02 = Treatment(absorptionRate=2.0, decayRate=0.02, dose=0.05, firstDoseTime=50.0, dosesNum=2, timeBetweenDoses=20.0)

    learnRandom = RandomSpace(
        timespaceDomain=timespace,
        initialSize=35*35,
        interiorSize=35*35*35,
        boundarySize=35
    )
    initialCondition = InitialCondition((60.0, 60.0), 0.4, 10)

    tracker = initializer.getTracker(timespace=timespace)

    pinn = initializer.getModel().to(device)

    weights = Weights(residual=1.0, initial=1.0, boundary=1.0)
    
    loss = Loss(
        learnRandom,
        initialCondition,
        diffusion,
        treatment00,
        weights,
    )
    trainer = Trainer(pinn, loss, tracker)
    trainer.train(learning_rate=0.002)
