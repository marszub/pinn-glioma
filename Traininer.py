from typing import Callable
import numpy as np
from Pinn import PINN
from Tracker import Tracker
import torch


class Trainer:
    def __init__(self, nn: PINN, loss: Callable, tracker: Tracker):
        self.nn = nn
        self.loss = loss
        self.tracker = tracker

    def train(
        self,
        learning_rate: int = 0.01,
        max_epochs: int = 1_000,
    ) -> PINN:
        optimizer = torch.optim.Adam(
            self.nn.parameters(), lr=learning_rate
        )
        self.tracker.start(self.loss.initial_condition)
        for _ in range(max_epochs):
            try:
                loss: torch.Tensor = self.loss(self.nn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.tracker.update(self.loss.verbose(self.nn), self.nn)
            except KeyboardInterrupt:
                self.tracker.finish(self.nn)
                break
