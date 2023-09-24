from copy import deepcopy
from typing import Callable
from menu.PauseMenu import PauseMenu
from Pinn import PINN
from tracking.Visualizer import Visualizer


class Tracker:
    def __init__(self, visualizer: Visualizer, epochs: int, isInteractive: bool, startPaused: bool):
        self.visualizer = visualizer
        self.maxEpochs = epochs
        self.__isInteractive = isInteractive
        self.__isTerminated = False
        self.__startPaused = startPaused

    def start(self, initialCondition: Callable, nn: PINN):
        self.epoch = 0
        self.bestApprox = deepcopy(nn)
        if self.__startPaused:
            self.pause()

    def update(self, lossValue: tuple, nn: PINN):
        self.epoch += 1

    def pause(self):
        if self.__isInteractive:
            pauseMenu = PauseMenu(self)
            pauseMenu.run()
            self.__isTerminated = pauseMenu.shouldTerminate()
        else:
            self.__isTerminated = True
    
    def isTraining(self):
        return not self.__isTerminated and self.epoch < self.maxEpochs

    def finish(self, pinn: PINN):
        pass
