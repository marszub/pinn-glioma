from os import path
from typing import Callable
import torch
from model.TimespaceDomain import TimespaceDomain
from plot.DataProvider import DataProvider


class SimulationLoader(DataProvider):
    def __init__(self, input_dir: str, timespace_domain: TimespaceDomain):
        super().__init__(timespace_domain)
        if input_dir is None or not path.isdir(input_dir):
            raise FileNotFoundError(f"Directory {input_dir} does not exist")
        self.input_dir = input_dir

    def for_each_frame(self, action: Callable):
        i = 0
        filepath = f"{self.input_dir}/sim_state_{i}.pt"
        while path.isfile(filepath):
            loaded_state = torch.load(filepath)
            t = loaded_state["time"]
            u = loaded_state["state"]
            t = t.to(self.device)
            u = u.to(self.device)
            action(t, u)
            i += 1
            filepath = f"{self.input_dir}/sim_state_{i}.pt"
        if i == 0:
            raise FileNotFoundError(
                f"No simulation frames found in {self.input_dir}.")
        print(f"Loaded {i} files.")

    def iterator(self):
        i = 0
        filepath = f"{self.input_dir}/sim_state_{i}.pt"
        while path.isfile(filepath):
            loaded_state = torch.load(filepath)
            t = loaded_state["time"]
            u = loaded_state["state"]
            t = t.to(self.device)
            u = u.to(self.device)
            yield t, u
            i += 1
            filepath = f"{self.input_dir}/sim_state_{i}.pt"
        if i == 0:
            raise FileNotFoundError(
                f"No simulation frames found in {self.input_dir}.")
        print(f"Loaded {i} files.")

    def get_sample_space(self):
        from pinn.simulationSpace.UniformSpace import UniformSpace

        u_shape = None
        time_resolution = 0
        for t, u in self.iterator():
            time_resolution += 1
            assert u_shape is None or u_shape == u.shape, f"{u_shape} {u.shape}"
            u_shape = u.shape
            assert u_shape[0] == u_shape[1], f"{u_shape}"
        return UniformSpace(
            self.timespace_domain,
            u.shape[1],
            time_resolution,
            requiresGrad=False,
        )

    def get_times(self):
        times = []
        for t, u in self.iterator():
            times.append(t)
        return times
