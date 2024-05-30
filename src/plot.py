#!/bin/python

from plot.ArgsParser import ArgsParser

if __name__ == "__main__":
    parser = ArgsParser()
    parser.show()
    args = parser.get()
    from model.Experiment import Experiment

    experiment = Experiment()

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
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )

    visualizer = Visualizer(
        plotter, space, args.output, args.plotTransparent
    )

    if args.plotType in parser.modelPlotTypes:
        from os import path

        if args.input is None:
            raise ValueError(
                f"Plotting {args.plotType} requires input to be speified")
        if path.isfile(args.input):
            from plot.PinnEvaluator import PinnEvaluator
            from pinn.PinnConfig import PinnConfig
            config = PinnConfig()
            data_provider = PinnEvaluator(args.input, space, config)
        elif path.isdir(args.input):
            from plot.SimulationLoader import SimulationLoader
            data_provider = SimulationLoader(
                args.input, experiment.timespaceDomain)
        else:
            print(f"File {args.input} does not exist")
            exit()

    diffusion = None
    if args.backgroundDiffusion:
        diffusion = experiment.getDiffusionMap()

    if args.plotType == "animation":
        visualizer.animateProgress(data_provider, args.fileName, diffusion)
    elif args.plotType == "ic":
        visualizer.plotIC(experiment.getInitialCondition(),
                          args.title, args.fileName, diffusion)
    elif args.plotType == "diffusion":
        visualizer.plotIC(experiment.getDiffusionMap(),
                          args.title, args.fileName)
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
        totalLoss = lossOverTime[:, :1]
        visualizer.plotLossMinMax(totalLoss, fileName=args.fileName)
    elif args.plotType == "sizeOverTime":
        times, sizes = data_provider.get_size_over_time()
        visualizer.plotSizeOverTime(times, sizes, args.fileName, args.title)
    elif args.plotType == "treatment":
        visualizer.plotTreatment(experiment.getTreatment(), args.fileName)
    else:
        print(f"Error: {args.plotType} plot type is not defined")
