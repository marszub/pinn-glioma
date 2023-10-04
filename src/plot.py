#!/bin/python

from model.Configuration import Configuration
from plotter.ArgsParser import ArgsParser
from plotter.Plotter3D import Plotter3D
from plotter.PlotterColor import PlotterColor
from simulationSpace.UniformSpace import UniformSpace
from plotter.Visualizer import Visualizer

if __name__ == "__main__":
    parser = ArgsParser()
    parser.show()
    args = parser.get()
    config = Configuration()

    if args.style == "color":
        plotter = PlotterColor()
    if args.style == "3d":
        plotter = Plotter3D()

    space = UniformSpace(
        timespaceDomain=config.getTimespaceDomain(),
        spaceResoultion=150,
        timeResoultion=20,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        plotter, space, args.output, args.plotTransparent
    )

    visualizer.printLoss([1, 2, 3, 4])
