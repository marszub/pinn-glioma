from SampleSpace import SampleSpace
from TimespaceDomain import TimespaceDomain
from Tracker import Tracker
import torch
from torch import nn
from Loss import Loss
from Pinn import PINN
from InitialCondition import initial_condition
from Traininer import Trainer
from Weights import Weights

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    timespace = TimespaceDomain(
        spaceDomains=[(0.0, 1.0), (0.0, 1.0)],
        timeDomain=(0.0, 1.0),
    )
    plotSpace = SampleSpace(
        timespaceDomain=timespace,
        spaceResoultion=150,
        timeResoultion=20,
    )
    learnSpace = SampleSpace(
        timespaceDomain=timespace,
        spaceResoultion=40,
        timeResoultion=40,
    )

    tracker = Tracker("./tmp", plotSpace)

    pinn = PINN(layers=4, neuronsPerLayer=200, act=nn.Tanh()).to(device)

    weights = Weights(residual=1.0, initial=1.0, boundary=1.0)
    loss = Loss(
        learnSpace,
        initial_condition,
        weights,
    )

    trainer = Trainer(pinn, loss, tracker)
    trainer.train(learning_rate=0.002, max_epochs=100_000)
