import asyncio
from typing import Callable
from pinn.Pinn import PINN
import torch


class Trainer:
    def __init__(self, initializer, loss: Callable):
        self.nn = initializer.getTrainModel()
        self.loss = loss
        self.initializer = initializer

    async def train(self) -> PINN:
        bestModel = self.initializer.getBestModel()
        tracker = self.initializer.getTracker()
        optimizer = self.initializer.getOptimizer()
        tracker.start(self.loss.validate(bestModel), bestModel)
        while tracker.isTraining():
            try:
                self.nn.train()
                losses: tuple = self.loss.verbose(self.nn)
                losses_copy = tuple(
                    [loss.detach().clone() for loss in losses])
                optimizer.zero_grad()
                losses[0].backward()
                optimizer.step()
                self.nn.eval()

                await asyncio.sleep(0)
                tracker.update(
                    losses_copy + (self.loss.validation(self.nn),),
                    self.nn, optimizer,
                )
            except KeyboardInterrupt:
                tracker.terminate()
