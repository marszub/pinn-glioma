from typing import Callable
from Pinn import PINN
from tracking.DefaultTracker import Tracker
import torch


class Trainer:
    def __init__(self, nn: PINN, loss: Callable, tracker: Tracker):
        self.nn = nn
        self.loss = loss
        self.tracker = tracker

    def train(
        self,
        learning_rate: int = 0.005,
    ) -> PINN:
        optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=learning_rate
        )
        self.tracker.start(self.loss.initialCondition, self.nn)
        while self.tracker.isTraining():
            try:
                self.nn.train()
                loss: torch.Tensor = self.loss(self.nn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.nn.eval()

                self.tracker.update(self.loss.verbose(self.nn), self.nn)
            except KeyboardInterrupt:
                self.nn.eval()
                self.tracker.pause()

        self.tracker.finish(self.nn)
