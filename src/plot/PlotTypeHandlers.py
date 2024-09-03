from pinn.simulationSpace.UniformSpace import UniformSpace
from model.Experiment import Experiment
from model.ExperimentLoader import ExperimentLoader


def load_model(model_path: str, space: UniformSpace, experiment: Experiment):
    from os import path

    if path.isfile(model_path):
        from plot.PinnEvaluator import PinnEvaluator
        return PinnEvaluator(model_path, space)
    elif path.isdir(model_path):
        from plot.SimulationLoader import SimulationLoader
        return SimulationLoader(
            model_path, experiment.timespaceDomain)
    print(f"File {model_path} does not exist")
    exit()


def load_comparable_models(
    model1_path: str,
    model2_path: str,
    space: UniformSpace,
    experiment: Experiment
):
    from os import path
    from plot.SimulationLoader import SimulationLoader
    from plot.PinnEvaluator import PinnEvaluator

    model1 = None
    model2 = None
    times = None

    if path.isdir(model1_path):
        print("model1 is simulation")
        model1 = SimulationLoader(model1_path, experiment.timespaceDomain)
        space, times = model1.get_sample_space_and_times()

    if path.isdir(model2_path):
        print("model2 is simulation")
        model2 = SimulationLoader(model2_path, experiment.timespaceDomain)
        space, times = model2.get_sample_space_and_times()

    if path.isfile(model1_path):
        print("model1 is pinn")
        model1 = PinnEvaluator(model1_path, space, times)

    if path.isfile(model2_path):
        print("model2 is pinn")
        model2 = PinnEvaluator(model2_path, space, times)

    assert model1 is not None and model2 is not None
    return model1, model2


def plot_animation(args):
    from plot.Plotter import Plotter
    from plot.Visualizer import Visualizer

    experiment = ExperimentLoader(args.experiment)
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

    experiment = ExperimentLoader(args.experiment)
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
        experiment.ic,
        plotter,
        diffusion
    )


def plot_size_over_time(args):
    from plot.Visualizer import Visualizer

    experiment = ExperimentLoader(args.experiment)

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
    visualizer.plotSizeOverTime(times, sizes, y_title="Tumor size")


def plot_difference(args):
    from plot.Visualizer import Visualizer
    import torch

    experiment = ExperimentLoader(args.experiment)

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
    with torch.no_grad():
        model1_data, model2_data = load_comparable_models(
            args.model1, args.model2, space, experiment)
        diffs = []
        times = []
        for (t1, u1), (t2, u2) in zip(
                model1_data.iterator(), model2_data.iterator()):
            assert torch.isclose(t1, t2)
            u1 = u1.reshape((-1,))
            u2 = u2.reshape((-1,))
            diff = torch.sum(torch.abs(u1 - u2)) / torch.sum(u1)
            diffs.append(diff)
            times.append(t1)
            u_size = torch.numel(u1)
        print(u_size)
        diffs = (
            torch.tensor(diffs) * 100.0 /
            experiment.timespaceDomain.get_points_per_space_unit(u_size)
        )
        times = torch.tensor(times)
        visualizer.plotSizeOverTime(
            times, diffs, y_title="Tumor concentration difference [%]")


def plot_diffusion(args):
    from plot.Visualizer import Visualizer
    from plot.Plotter import Plotter

    experiment = ExperimentLoader(args.experiment)

    timeResolution = 20
    spaceResoultion = 300

    plotter = Plotter(cmap="bone")
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
        plotter,
    )


def plot_treatment(args):
    from plot.Visualizer import Visualizer

    experiment = ExperimentLoader(args.experiment)

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

    experiment = ExperimentLoader(args.experiment)

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
        labels=["Total", "Residual", "Initial",
                "Boundary", "Data", "Validation"],
    )


def plot_total_loss(args):
    from plot.Visualizer import Visualizer
    from pinn.Loader import loadMetrics
    import numpy as np

    experiment = ExperimentLoader(args.experiment)

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
