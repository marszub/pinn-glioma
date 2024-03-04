from copy import deepcopy
from model.Pinn import PINN
from train.Saver import Saver
from train.tracking.SharedData import SharedData


class Tracker:
    def __init__(
        self, modelSaver: Saver, epochs: int, sharedData: SharedData
    ):
        self.modelSaver = modelSaver
        self.maxEpochs = epochs
        self.sharedData = sharedData

    def start(self, lossValue: tuple, nn: PINN):
        self.epoch = 0
        self.bestLoss = lossValue
        self.bestApprox = deepcopy(nn)

    def update(self, lossValue: tuple, nn: PINN):
        self.epoch += 1

    def lateUpdate(self):
        if self.sharedData.save:
            self.sharedData.save = False
            self.modelSaver.saveModel(self.bestApprox)

    def terminate(self):
        self.sharedData.terminate = True

    def isTraining(self):
        return (
            not self.sharedData.terminate and self.epoch < self.maxEpochs
        )
