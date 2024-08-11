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
            "-d",
            "--data-dir",
            help="Path to dir storing training data set. The dir should contain ONLY training data files. Each file is a dict saved by pytorch containing 'time' and 'state' records.",
            type=str,
            default=None,
            action="store",
            dest="data_dir",
        )
        self.parser.add_argument(
            "-v",
            "--validation-dir",
            help="Path to dir storing validation data set. The dir should contain ONLY training data files. Each file is a dict saved by pytorch containing 'time' and 'state' records.",
            type=str,
            default=None,
            action="store",
            dest="validation_dir",
        )
        self.parser.add_argument(
            "-l",
            "--load",
            help="Load previously trained model.",
            action="store",
        )
        self.config = self.parser.parse_args()
        self.config.data_dir = self.__trim_dir(self.config.data_dir)
        self.config.validation_dir = self.__trim_dir(self.config.validation_dir)

    def __trim_dir(self, dir_name):
        if dir_name is not None and dir_name.endswith('/'):
            return dir_name[:-1]
        return dir_name


    def get(self):
        return self.config

    def show(self):
        print(self.config)
