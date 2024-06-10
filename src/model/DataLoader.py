import torch
import os
import re


class DataLoader:
    def __init__(self, dir_name: str):
        self.dir_name = dir_name

    def get_indices(self):
        filenames = filter(re.compile(
            "sim_state_(0|([1-9][0-9]*)).pt").match, os.listdir(self.dir_name))
        return [int(re.findall(r"\d+", filename)[0]) for filename in filenames]

    def get_data(self, indices):
        ts = []
        us = []
        for i in indices:
            filepath = f"{self.dir_name}/sim_state_{i}.pt"
            assert os.path.isfile(filepath)
            loaded_state = torch.load(filepath)
            ts.append(loaded_state["time"])
            us.append(loaded_state["state"])
            i += 1
        print(f"Loaded {len(indices)} simulation data frames.")

        return torch.stack(us, axis=-1), torch.stack(ts)
