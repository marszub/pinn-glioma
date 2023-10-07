import asyncio
from typing import Callable
from model.Pinn import PINN
from train.tracking.DefaultTracker import Tracker
import torch


class Trainer:
    def __init__(
        self, nn: PINN, loss: Callable, tracker: Tracker, learningRate
    ):
        self.nn = nn
        self.loss = loss
        self.tracker = tracker
        self.learningRate = learningRate

    async def train(self) -> PINN:
        optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=self.learningRate
        )
        self.tracker.start(self.nn)
        while self.tracker.isTraining():
            try:
                self.nn.train()
                loss: torch.Tensor = self.loss(self.nn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.nn.eval()

                await asyncio.sleep(0)
                self.tracker.update(self.loss.verbose(self.nn), self.nn)
            except KeyboardInterrupt:
                self.tracker.terminate()
