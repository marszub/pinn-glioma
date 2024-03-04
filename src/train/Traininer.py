import asyncio
from typing import Callable
from model.Pinn import PINN
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
        tracker.start(self.loss.verbose(bestModel), bestModel)
        while tracker.isTraining():
            try:
                self.nn.train()
                loss: torch.Tensor = self.loss(self.nn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.nn.eval()

                await asyncio.sleep(0)
                tracker.update(self.loss.verbose(self.nn), self.nn)
            except KeyboardInterrupt:
                tracker.terminate()
