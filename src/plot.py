#!/bin/python

from plot.ArgsParser import ArgsParser

if __name__ == "__main__":
    parser = ArgsParser()
    parser.show()
    args = parser.get()
    from model.Configuration import Configuration

    config = Configuration()

    from plot.Plotter import Plotter
    plotter = Plotter(limit=args.maxU, cmap=args.cmap)

    timeResolution = 20
    spaceResoultion = 300
    if args.plotType == "sizeOverTime":
        timeResolution = 50
        spaceResoultion = 150

    from pinn.simulationSpace.UniformSpace import UniformSpace
    from pinn.Loader import loadMetrics
    from plot.Visualizer import Visualizer
    import numpy as np

    space = UniformSpace(
        timespaceDomain=config.getTimespaceDomain(),
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )

    visualizer = Visualizer(
        plotter, space, args.output, args.plotTransparent
    )

    if args.plotType in parser.modelPlotTypes:
        from torch import load
        from os import path

        model = config.getNeuralNetwork()
        if args.input is not None and path.isfile(args.input):
            model.load_state_dict(load(args.input))
            model.eval()
        else:
            print(f"File {args.input} does not exist")
            exit()
    
    diffusion = None
    if args.backgroundDiffusion:
        diffusion = config.getDiffusionMap()

    if args.plotType == "animation":
        visualizer.animateProgress(model, args.fileName, diffusion)
    elif args.plotType == "ic":
        visualizer.plotIC(config.getInitialCondition(), args.title, args.fileName, diffusion)
    elif args.plotType == "diffusion":
        visualizer.plotIC(config.getDiffusionMap(), args.title, args.fileName)
    elif args.plotType == "loss":
        lossOverTime = np.array(loadMetrics(args.input), dtype=float)
        if lossOverTime is None:
            print("Failed to load")
            exit()
        visualizer.plotLosses(
            loss_over_time=lossOverTime,
            fileName=args.fileName,
            labels=["Total", "Residual", "Initial", "Boundary"],
        )
    elif args.plotType == "totalLoss":
        lossOverTime = np.array(loadMetrics(args.input), dtype=float)
        if lossOverTime is None:
            print("Failed to load")
            exit()
        totalLoss = lossOverTime[:,:1]
        visualizer.plotLossMinMax(totalLoss, fileName=args.fileName)
    elif args.plotType == "sizeOverTime":
        visualizer.plotSizeOverTime(model, args.fileName)
    elif args.plotType == "treatment":
        visualizer.plotTreatment(config.getTreatment(), args.fileName)
    else:
        print(f"Error: {args.plotType} plot type is not defined")
