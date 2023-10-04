from argparse import ArgumentParser


class ArgsParser:
    def __init__(self):
        self.parser = ArgumentParser(
            prog="plot",
            description="Program visualizing trained model, loss value during training and given conditions. ",
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
            help="Choose style of generated plots. (default: %(default)s)",
            choices=["color", "3d"],
            action="store",
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
        self.config = self.parser.parse_args()

    def get(self):
        return self.config

    def show(self):
        print(self.config)


if __name__ == "__main__":
    menu = ArgsParser()
    menu.show()
