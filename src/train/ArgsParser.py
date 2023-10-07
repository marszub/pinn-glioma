from argparse import ArgumentParser


class ArgsParser:
    def __init__(self):
        self.parser = ArgumentParser(
            prog="pinn",
            description="Program to train pinn.",
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
            help="Allows saving model during training with 's' key.",
            action="store_true",
        )
        self.parser.add_argument(
            "-l",
            "--load",
            help="Load previously trained model.",
            action="store",
        )
        self.parser.add_argument(
            "-s",
            "--simpleOutput",
            help="Only model and loss chart are saved at the end of training.",
            action="store_true",
        )
        self.config = self.parser.parse_args()

    def get(self):
        return self.config

    def show(self):
        print(self.config)
