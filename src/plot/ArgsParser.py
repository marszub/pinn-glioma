from argparse import ArgumentParser


class ArgsParser:
    def __init__(self):
        self.modelPlotTypes = ["animation", "sizeOverTime"]
        self.conditionPlotTypes = ["ic", "treatment"]
        self.otherPlotTypes = ["loss"]
        self.plotTypes = list(
            set(self.modelPlotTypes + self.conditionPlotTypes + self.otherPlotTypes)
        )
        self.plotTypes.sort()
        self.parser = ArgumentParser(
            prog="plot",
            description="Program visualizing trained model, loss value during training and given conditions. ",
        )
        self.parser.add_argument(
            "plotType",
            help="Plot type. One of: %(choices)s",
            type=str,
            choices=self.plotTypes,
            action="store",
            metavar="type",
        )
        self.parser.add_argument(
            "-o",
            "--output",
            default="tmp",
            help="Path to output dir where plots are saved. (default: %(default)s)",
            metavar="DIR",
        )
        self.parser.add_argument(
            "-s",
            "--style",
            default="color",
            help="Choose style of generated plots. One of: %(choices)s. (default: %(default)s)",
            choices=["color", "3d"],
            action="store",
            metavar="STYLE",
        )
        self.parser.add_argument(
            "--cmap",
            default="RdPu",
            help="Choose style of generated plots. One of: %(choices)s. (default: %(default)s)",
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
        self.parser.add_argument(
            "--maxU",
            help="Set maximum value on Z axis (tumor concentration) on drawn space plots. If set to None, each plot will be adapted. (default: %(default)s)",
            type=float,
            action="store",
        )
        self.parser.add_argument(
            "--plotTransparent",
            help="If set, all plots will have transparent background. (default: %(default)s)",
            action="store_true",
        )
        self.parser.add_argument(
            "input",
            help="Input path.",
            type=str,
            action="store",
        )
        self.parser.add_argument(
            "name",
            help="Name of the plot",
            type=str,
            action="store",
        )
        self.config = self.parser.parse_args()

    def get(self):
        return self.config

    def show(self):
        print(self.config)
