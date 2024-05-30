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
            action(t, u)
            i += 1
            filepath = f"{self.input_dir}/sim_state_{i}.pt"
        if i == 0:
            raise FileNotFoundError(f"No simulation frames found in {self.input_dir}.")
        print(f"Loaded {i} files.")
