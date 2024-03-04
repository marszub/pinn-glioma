from copy import deepcopy
from time import time
from model.Pinn import PINN
from train.Saver import Saver

class Tracker:
    def __init__(self, modelSaver: Saver, epochs: int):
        self.modelSaver = modelSaver
        self.maxEpochs = epochs
        self.lossValues = []
        self.bestLoss = [float("inf") for _ in range(4)]
        self.isTerminated = False

    def start(self, lossValue: tuple, nn: PINN):
        self.epoch = 0
        self.bestLoss = lossValue
        self.bestApprox = deepcopy(nn)
        self.epochStartTime = time()

    def update(self, lossValue: tuple, nn: PINN):
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
