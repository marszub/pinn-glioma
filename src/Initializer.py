from os import path
from Pinn import PINN
from simulationSpace.TimespaceDomain import TimespaceDomain
from tracking.DefaultTracker import DefaultTracker, Tracker
from simulationSpace.UniformSpace import UniformSpace
from plotter.Plotter3D import Plotter3D
from plotter.PlotterColor import PlotterColor
from tracking.SimpleTracker import SimpleTracker
from tracking.Visualizer import Visualizer
from torch import load, nn


class Initializer:
    def __init__(self, config):
        self.config = config

    def getTracker(self, timespace: TimespaceDomain) -> Tracker:
        plotSpace = UniformSpace(
            timespaceDomain=timespace,
            spaceResoultion=150,
            timeResoultion=20,
            requiresGrad=False,
        )
        if self.config.plot == "3d":
            plotter = Plotter3D(limit=self.config.plotMax)
        else:
            plotter = PlotterColor(limit=self.config.plotMax)

        visualizer = Visualizer(
            plotter,
            plotSpace,
            self.config.output,
            self.config.plotTransparent,
        )
        if self.config.simpleOutput:
            return SimpleTracker(
                visualizer,
                epochs=self.config.epochs,
                isInteractive=self.config.interactive,
                startPaused=self.config.pause,
            )

        return DefaultTracker(
            visualizer,
            epochs=self.config.epochs,
            isInteractive=self.config.interactive,
            startPaused=self.config.pause,
        )

    def initialize(self, model):
        if self.config.load is not None and path.isfile(self.config.load):
            model.load_state_dict(load(self.config.load))
            model.eval()
        return model
