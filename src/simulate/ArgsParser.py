from argparse import ArgumentParser


class ArgsParser:
    def __init__(self):
        self.parser = ArgumentParser(
            prog="simulate",
            description="Program to predict glioma growth using FDM",
        )

        self.parser.add_argument(
            "-o",
            "--output",
            default="tmp/simulation",
            help="Path to output dir. (default: %(default)s)",
            metavar="DIR",
        )
        self.parser.add_argument(
            "--experiment",
            default=None,
            help="Path to python script that overrides default Experiment definition. " +
            "(default: %(default)s)",
            type=str,
            action="store",
            metavar="PATH",
        )
        exclusive = self.parser.add_mutually_exclusive_group()
        exclusive.add_argument(
            "-l",
            "--load",
            help="Resume simulation form state stored in specified file.",
            action="store",
            metavar="FILE",
        )
        exclusive.add_argument(
            "-r",
            "--spatial-resolution",
            default=200,
            help="Number of points in each spatial dimention. (default: %(default)s)",
            type=int,
            action="store",
            metavar="POINTS",
        )
        self.parser.add_argument(
            "-t",
            "--time-resolution",
            default=1000,
            help="Number of time steps in the simulation. (default: %(default)s)",
            type=int,
            action="store",
            metavar="POINTS",
        )
        self.parser.add_argument(
            "-s",
            "--silent",
            help="Run in silent mode. (default: %(default)s)",
            action="store_true",
        )
        self.config = self.parser.parse_args()

    def get(self):
        return self.config

    def show(self):
        print(self.config)
