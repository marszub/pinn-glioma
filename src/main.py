#!/bin/python

from Initializer import Initializer
from menu.ArgsParser import ArgsParser
from model.Configuration import Configuration
import torch
from loss.Loss import Loss
from Traininer import Trainer
from Weights import Weights
from simulationSpace.RandomSpace import RandomSpace

if __name__ == "__main__":
    config = Configuration()

    argsParser = ArgsParser()
    argsParser.show()
    args = argsParser.get()
    initializer = Initializer(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    tracker = initializer.getTracker(timespace=config.getTimespaceDomain())

    pinn = initializer.initialize(config.getNeuralNetwork()).to(device)

    learnSpace = RandomSpace(
        timespaceDomain=config.getTimespaceDomain(),
        initialSize=35 * 35,
        interiorSize=35 * 35 * 35,
        boundarySize=35,
    )

    weights = Weights(residual=1.0, initial=1.0, boundary=1.0)

    loss = Loss(
        learnSpace,
        config.getInitialCondition(),
        config.getDiffusionMap(device),
        config.getTreatment(),
        weights,
    )
    trainer = Trainer(pinn, loss, tracker)
    trainer.train(learning_rate=0.002)
