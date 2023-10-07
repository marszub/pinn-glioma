from os import path
from train.InteractionManager import InteractionManager
from train.Traininer import Trainer
from train.Threads import interactiveTrainingThread, trainingThread
from train.tracking.DefaultTracker import DefaultTracker, Tracker
from train.tracking.SharedData import SharedData
from train.tracking.SilentTracker import SilentTracker
from torch import load
from train.ModelSaver import ModelSaver


class Initializer:
    def __init__(self, config):
        self.config = config
        self.sharedData = SharedData()

    def getTracker(self) -> Tracker:
        modelSaver = ModelSaver(self.config.output)

        if self.config.silentOutput:
            return SilentTracker(
                modelSaver,
                epochs=self.config.epochs,
                sharedData=self.sharedData,
            )

        return DefaultTracker(
            modelSaver,
            epochs=self.config.epochs,
            sharedData=self.sharedData,
        )

    def initialize(self, model):
        if self.config.load is not None and path.isfile(self.config.load):
            model.load_state_dict(load(self.config.load))
            model.eval()
        return model

    def getMainThread(self, trainer: Trainer):
        if self.config.interactive:
            return interactiveTrainingThread(
                trainer, InteractionManager(self.sharedData)
            )
        return trainingThread(trainer)
