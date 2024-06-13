from pinn.simulationSpace.UniformSpace import UniformSpace
from model.Experiment import Experiment


def load_model(model_path: str, space: UniformSpace, experiment: Experiment):
    from os import path

    if path.isfile(model_path):
        from plot.PinnEvaluator import PinnEvaluator
        from pinn.PinnConfig import PinnConfig
        config = PinnConfig()
        return PinnEvaluator(model_path, space, config)
    elif path.isdir(model_path):
        from plot.SimulationLoader import SimulationLoader
        return SimulationLoader(
            model_path, experiment.timespaceDomain)
    print(f"File {model_path} does not exist")
    exit()


def plot_animation(args):
    from plot.Plotter import Plotter
    from plot.Visualizer import Visualizer

    experiment = Experiment()
    plotter = Plotter(limit=args.max_u, cmap=args.cmap)
    diffusion = None
    if args.background_diffusion:
        diffusion = experiment.diffusion

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    data_provider = load_model(args.input, space, experiment)
    visualizer.animateProgress(
        data_provider, plotter, diffusion)


def plot_initial_condition(args):
    from plot.Plotter import Plotter
    from plot.Visualizer import Visualizer

    experiment = Experiment()
    plotter = Plotter(limit=args.max_u, cmap=args.cmap)
    diffusion = None
    if args.background_diffusion:
        diffusion = experiment.diffusion

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    visualizer.plotIC(
        experiment.getInitialCondition(),
        plotter,
        diffusion
    )


def plot_size_over_time(args):
    from plot.Visualizer import Visualizer

    experiment = Experiment()

    timeResolution = 50
    spaceResoultion = 150

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    data_provider = load_model(args.input, space, experiment)
    times, sizes = data_provider.get_size_over_time()
    visualizer.plotSizeOverTime(times, sizes)


def plot_difference(args):
    from plot.Visualizer import Visualizer

    experiment = Experiment()

    timeResolution = 50
    spaceResoultion = 150

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    model1_data = load_model(args.model1, space, experiment)
    model2_data = load_model(args.model2, space, experiment)
    diffs = []
    for (t1, u1), (t2, u2) in zip(model1_data.iterator(), model2_data.iterator()):
        # TODO: Make sure this works
        assert t1 == t2
        diff = torch.sum(torch.abs(u1 - u2))
        diffs.append(diff)
    diffs = torch.tensor(diffs)
    # TODO: Dedicated plot function. Also needs to be tested
    visualizer.plotSizeOverTime(times, sizes)


def plot_diffusion(args):
    from plot.Visualizer import Visualizer

    experiment = Experiment()

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    visualizer.plotIC(
        experiment.diffusion,
    )


def plot_treatment(args):
    from plot.Visualizer import Visualizer

    experiment = Experiment()

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    visualizer.plotTreatment(experiment.treatment)


def plot_loss(args):
    from plot.Visualizer import Visualizer
    from pinn.Loader import loadMetrics
    import numpy as np

    experiment = Experiment()

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    lossOverTime = np.array(loadMetrics(args.input), dtype=float)
    if lossOverTime is None:
        print("Failed to load")
        exit()
    visualizer.plotLosses(
        loss_over_time=lossOverTime,
        labels=["Total", "Residual", "Initial", "Boundary", "Data"],
    )


def plot_total_loss(args):
    from plot.Visualizer import Visualizer
    from pinn.Loader import loadMetrics
    import numpy as np

    experiment = Experiment()

    timeResolution = 20
    spaceResoultion = 300

    space = UniformSpace(
        timespaceDomain=experiment.timespaceDomain,
        spaceResoultion=spaceResoultion,
        timeResoultion=timeResolution,
        requiresGrad=False,
    )
    visualizer = Visualizer(
        space, args.output, args.title, args.plot_transparent
    )
    lossOverTime = np.array(loadMetrics(args.input), dtype=float)
    if lossOverTime is None:
        print("Failed to load")
        exit()
    totalLoss = lossOverTime[:, :1]
    visualizer.plotLossMinMax(totalLoss)
