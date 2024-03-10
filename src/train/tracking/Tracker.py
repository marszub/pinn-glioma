from copy import deepcopy
from time import time
from model.Pinn import PINN
from train.Saver import Saver

class Tracker:
    def __init__(self, epochs: int, epoch: int, lossValues: list):
        self.maxEpochs = epochs
        self.lossValues = lossValues
        self.isTerminated = False
        self.epoch = epoch

    def start(self, lossValue: tuple, nn: PINN):
        self.bestLoss = list(map(lambda element: element.item(), lossValue))
        self.bestApprox = deepcopy(nn)
        self.epochStartTime = time()

    def update(self, lossValue: tuple, nn: PINN, optimizer):
        lossValue = list(map(lambda element: element.item(), lossValue))
        self.lossValues.append(lossValue)
        self.epoch += 1
        totalLoss = lossValue[0]
        if (self.epoch) % 1000 == 0:
            print(
                f"Epoch: {self.epoch}\tLoss: {float(totalLoss):>7f}\t"
                + f"Time: {time() - self.epochStartTime:.1f}s"
            )
            self.epochStartTime = time()

        if totalLoss < self.bestLoss[0]:
            self.bestApprox = deepcopy(nn)
            self.bestLoss = lossValue

    def terminate(self):
        self.isTerminated = True

    def isTraining(self):
        return not self.isTerminated and self.epoch < self.maxEpochs
