from argparse import ArgumentParser


class ArgsParser:
    def __init__(self):
        self.parser = ArgumentParser(
            prog="pinn",
            description="Program to train, evaluate and visalize pinn.",
        )
        self.parser.add_argument(
            "-e",
            "--epochs",
            help="Maximum number of training epochs. (default: %(default)s)",
            type=int,
            default=50_000,
            action="store",
        )
        self.parser.add_argument(
            "-o",
            "--output",
            default="tmp",
            help="Path to output dir. (default: %(default)s)",
            metavar="DIR",
        )
        self.parser.add_argument(
            "-i",
            "--interactive",
            help="Access menu with ctrl+C during training. Training is paused and can be resumed. ",
            action="store_true",
        )
        self.parser.add_argument(
            "-l",
            "--load",
            help="Load previously trained model.",
            action="store",
        )
        self.parser.add_argument(
            "--pause",
            help="Immidiatelly shows pause menu. Automaticly enables --interactive option. ",
            action="store_true",
        )
        self.parser.add_argument(
            "-p",
            "--plot",
            default="color",
            help="Choose style of generated plots. (default: %(default)s)",
            choices=["color", "3d"],
            action="store",
        )
        self.parser.add_argument(
            "-s",
            "--simpleOutput",
            help="Only model and loss chart are saved at the end of training.",
            action="store_true",
        )
        self.parser.add_argument(
            "--plotMax",
            help="Set maximum value on Z axis (tumor concentration) on drawn plots. If set to None, each plot will be adapted. (default: %(default)s)",
            type=float,
            action="store",
        )
        self.parser.add_argument(
            "--plotTransparent",
            help="If set, generated plots will have transparent background. (default: %(default)s)",
            action="store_true",
        )
        self.config = self.parser.parse_args()
        if self.config.pause:
            self.config.interactive = True

    def get(self):
        return self.config

    def show(self):
        print(self.config)


if __name__ == "__main__":
    menu = ArgsParser()
    menu.show()
