#!/bin/python

from os import path
from torch import load
from model.Configuration import Configuration
from plot.ArgsParser import ArgsParser
from plot.Plotter3D import Plotter3D
from plot.PlotterColor import PlotterColor
from simulationSpace.UniformSpace import UniformSpace
from plot.Visualizer import Visualizer

if __name__ == "__main__":
    parser = ArgsParser()
    parser.show()
    args = parser.get()
    config = Configuration()

    if args.style == "color":
        plotter = PlotterColor(limit=args.maxU)
    if args.style == "3d":
        plotter = Plotter3D(limit=args.maxU)

    timeResolution = 20
    if args.plotType == "sizeOverTime":
        timeResolution = 150

    space = UniformSpace(
        timespaceDomain=config.getTimespaceDomain(),
        spaceResoultion=150,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        plotter, space, args.output, args.plotTransparent
    )

    if args.plotType in parser.modelPlotTypes:
        model = config.getNeuralNetwork()
        if path.isfile(args.input):
            model.load_state_dict(load(args.input))
            model.eval()

    if args.plotType == "animation":
        visualizer.animateProgress(model, args.name)
    elif args.plotType == "sizeOverTime":
        visualizer.plotSizeOverTime(model, args.name)
    elif args.plotType == "ic":
        visualizer.plotIC(config.getInitialCondition(), args.name)
    elif args.plotType == "treatment":
        visualizer.plotTreatment(config.getTreatment(), args.name)
