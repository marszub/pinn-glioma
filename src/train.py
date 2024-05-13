#!/bin/python

import asyncio
from train.Initializer import Initializer
from train.ArgsParser import ArgsParser
from pinn.PinnConfig import PinnConfig
import torch
from pinn.loss.Loss import Loss
from model.Experiment import Experiment
from train.Traininer import Trainer
from pinn.simulationSpace.RandomSpace import RandomSpace

if __name__ == "__main__":
    config = PinnConfig()
    experiment = Experiment()

    argsParser = ArgsParser()
    argsParser.show()
    args = argsParser.get()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    initializer = Initializer(args, config, device)

    learnSpace = RandomSpace(
        timespaceDomain=experiment.timespaceDomain,
        initialSize=35 * 35,
        interiorSize=35 * 35 * 35,
        boundarySize=35,
    )

    loss = Loss(
        learnSpace,
        experiment,
        config.weights,
    )
    trainer = Trainer(initializer, loss)
    asyncio.run(initializer.getMainThread(trainer))
