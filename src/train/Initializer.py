from copy import deepcopy
from train.InteractionManager import InteractionManager
from train.Traininer import Trainer
from train.Threads import interactiveTrainingThread, trainingThread
from train.tracking.AutosaveTracker import Tracker
from train.tracking.SharedData import SharedData
from train.Saver import Saver
from pinn.PinnConfig import PinnConfig
import torch


class Initializer:
    def __init__(self, runConfig, experiment, device):
        self.device = device
        self.pinn_config = PinnConfig(
            experiment,
            runConfig.data_dir,
            runConfig.validation_dir,
        )
        self.runConfig = runConfig
        self.epoch = 0
        self.lossOverTime = []
        if self.runConfig.load is not None:
            self.__loadState()
        else:
            from pinn.Pinn import PINN
            self.trainModel = PINN(
                layers=runConfig.layers,
                neuronsPerLayer=runConfig.neurons,
            ).to(device)
            self.bestModel = deepcopy(self.trainModel).to(device)
            self.optimizer = torch.optim.Adam(self.trainModel.parameters())
        self.sharedData = SharedData()

    def __loadState(self):
        from pinn.Loader import loadTrainState

        state = loadTrainState(self.runConfig.load, self.device)
        self.epoch = state["epoch"]
        self.bestModel = state["bestModel"]
        self.trainModel = state["model"]
        self.optimizer = state["optimizer"]
        self.lossOverTime = state["lossOverTime"]

    def getTracker(self) -> Tracker:
        modelSaver = Saver(self.runConfig.output)
        if self.runConfig.interactive:
            from train.tracking.InteractiveTracker import (
                InteractiveTracker,
            )

            return InteractiveTracker(
                modelSaver,
                epochs=self.runConfig.epochs,
                epoch=self.epoch,
                lossValues=self.lossOverTime,
                sharedData=self.sharedData,
            )
        from train.tracking.AutosaveTracker import AutosaveTracker

        return AutosaveTracker(modelSaver, epochs=self.runConfig.epochs, epoch=self.epoch, lossValues=self.lossOverTime)

    def getTrainModel(self):
        return self.trainModel

    def getBestModel(self):
        return self.bestModel

    def getOptimizer(self):
        return self.optimizer

    def getPinnConfig(self):
        return self.pinn_config

    def getMainThread(self, trainer: Trainer):
        if self.runConfig.interactive:
            return interactiveTrainingThread(
                trainer, InteractionManager(self.sharedData)
            )
        return trainingThread(trainer)
