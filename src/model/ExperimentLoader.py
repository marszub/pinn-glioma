from os import path
import importlib.util


class ExperimentLoader:
    def __new__(cls, experiment_path):
        if experiment_path is None:
            from model.Experiment import Experiment
            return Experiment()
        assert path.isfile(experiment_path)
        spec = importlib.util.spec_from_file_location(
            "Experiment", experiment_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Experiment()
