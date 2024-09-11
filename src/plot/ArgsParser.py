from argparse import ArgumentParser
from plot.PlotTypeHandlers import (
    plot_animation,
    plot_initial_condition,
    plot_size_over_time,
    plot_difference,
    plot_treatment,
    plot_diffusion,
    plot_loss,
    plot_total_loss,
)


class ArgsParser:
    def __init__(self):
        self.plotTypes = ["animation", "size-over-time", "difference",
                          "initial-condition", "treatment", "diffusion", "loss", "total-loss"]
        self.plotTypes.sort()
        tumor_plot_parser = ArgumentParser(add_help=False)
        tumor_plot_parser.add_argument(
            "--cmap",
            default="RdPu",
            help="Choose color style of generated plots. " +
            "One of: %(choices)s. " +
            "(default: %(default)s)",
            type=str,
            choices=[
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "Greys",
                "Purples",
                "Blues",
                "Greens",
                "Oranges",
                "Reds",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
                "YlGnBu",
                "PuBuGn",
                "BuGn",
                "YlGn",
            ],
            action="store",
            metavar="CMAP",
        )
        tumor_plot_parser.add_argument(
            "--max-u",
            help="Set maximum value on Z axis (tumor concentration) on drawn space plots. " +
            "If set to None, each plot will be adapted. (default: %(default)s)",
            type=float,
            action="store",
            dest="max_u",
        )
        tumor_plot_parser.add_argument(
            "--background-diffusion",
            help="Plot diffusion map in the background of tumor plots. " +
            "(default: %(default)s)",
            action="store_true",
            dest="background_diffusion",
        )
        single_model_parser = ArgumentParser(add_help=False)
        single_model_parser.add_argument(
            "input",
            help="Input path to ploted model or simulation. " +
            "If path is file then plotting NN is assumed. " +
            "If path is directory, plotting simulation is assumed.",
            type=str,
            action="store",
        )
        compare_parser = ArgumentParser(add_help=False)
        compare_parser.add_argument(
            "--train-data",
            help="Path to training data to mark on the plot.",
            type=str,
            action="store",
            default=None,
            dest="train_data",
        )
        loss_parser = ArgumentParser(add_help=False)
        loss_parser.add_argument(
            "input",
            help="Input path to text file with table of loss values during training. ",
            type=str,
            action="store",
        )

        parser = ArgumentParser(
            prog="plot.py",
            description="Program visualizing trained model, " +
            "loss value during training and given conditions. ",
        )
        parser.add_argument(
            "-o",
            "--output",
            default="tmp/figure",
            help="Prefix of plots' file names (Eg. '/home/user/plots/figure'). " +
            "Actual file name is created by adding sufix at the end of provided string. " +
            "The sufix depends on plot type. Can be only file extension (Eg. '.gif')" +
            " or indentifier and file extension (Eg. '_11.png'). " +
            "(default: %(default)s)",
            type=str,
            action="store",
            metavar="PREFIX",
        )
        parser.add_argument(
            "--title",
            help="Title of the plot shown above the plot. " +
            "Can be provided multiple times to make multiline title. ",
            type=str,
            action="append",
            metavar="TITLE",
            default=[],
        )
        parser.add_argument(
            "--plot-transparent",
            help="If set, all plots will have transparent background. " +
            "(default: %(default)s)",
            action="store_true",
            dest="plot_transparent",
        )
        parser.add_argument(
            "--experiment",
            default=None,
            help="Path to python script that overrides default Experiment definition. " +
            "(default: %(default)s)",
            type=str,
            action="store",
            metavar="PATH",
        )
        subparsers = parser.add_subparsers(
            title='Plot types',
            dest='plot_type',
            metavar='TYPE',
        )
        subparsers.add_parser(
            'animation',
            aliases=['anim'],
            help="Create gif with animation of tumor shape over time.",
            parents=[tumor_plot_parser, single_model_parser],
        ).set_defaults(handler=plot_animation)
        sot_parser = subparsers.add_parser(
            'size_over_time',
            aliases=['sot'],
            help="Tumor size over time generated by model or simulation.",
            parents=[single_model_parser, compare_parser],
        )
        sot_parser.add_argument(
            "--other-model",
            help="Path to simulation plotted as GT",
            type=str,
            action="store",
            default=None,
            dest="other_model",
        )
        sot_parser.set_defaults(handler=plot_size_over_time)
        subparsers.add_parser(
            'initial_condition',
            aliases=['ic'],
            help="2D plot visualizing real tumor density for t=0.",
            parents=[tumor_plot_parser],
        ).set_defaults(handler=plot_initial_condition)
        difference_parser = subparsers.add_parser(
            'difference',
            aliases=['diff'],
            help="Difference between two models " +
            "and/or simulations in tumor density over time. " +
            "Exactly two input paths must be provided. " +
            "Each path must point to a model or a simulation. " +
            "If path is a file then NN is assumed. " +
            "If path is a directory, simulation is assumed.",
            parents=[compare_parser],
        )
        difference_parser.add_argument(
            "model1",
            help="Path to the first model",
            type=str,
            action="store",
        )
        difference_parser.add_argument(
            "model2",
            help="Path to the second model",
            type=str,
            action="store",
        )
        difference_parser.set_defaults(handler=plot_difference)
        subparsers.add_parser(
            'tretment',
            aliases=['treat'],
            help="Assumed treatment factor value over time.",
        ).set_defaults(handler=plot_treatment)
        subparsers.add_parser(
            'diffusion',
            aliases=['D'],
            help="Assumed diffusion coeffitient value in space.",
        ).set_defaults(handler=plot_diffusion)
        subparsers.add_parser(
            'loss',
            help="Loss during training split into parts " +
            "(initial, boudary, residual, data) and the total loss.",
            parents=[loss_parser],
        ).set_defaults(handler=plot_loss)
        total_loss_parser = subparsers.add_parser(
            'total_loss',
            aliases=['tloss'],
            help="Averrage loss in intervals over training time. " +
            "Plot also shows the best fit and minimum and maximum loss value in each interval.",
            parents=[loss_parser],
        )
        total_loss_parser.add_argument(
            "--validation",
            help="Plot validation loss instead of total loss. (default: total)",
            action="store_const",
            dest="loss_idx",
            default=0,
            const=5,
        )
        total_loss_parser.set_defaults(handler=plot_total_loss)

        self.config = parser.parse_args()

    def get(self):
        return self.config

    def show(self):
        print(self.config)
