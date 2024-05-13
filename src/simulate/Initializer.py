import torch


class Initializer:
    def __init__(self, run_config, experiment):
        self.run_config = run_config
        self.experiment = experiment
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")
        if self.run_config.load is not None:
            self.__load_state()
        else:
            self.__init_state()

    def __load_state(self):
        from torch import load
        from simulate.Grid import Grid
        loaded_object = load(self.run_config.load)
        self.save_id = loaded_object["save_id"] + 1
        self.t = loaded_object["t"]
        self.state = loaded_object["state"]
        assert (self.state.shape[0] == self.state.shape[1])
        assert (self.state.shape[0] > 2)
        self.grid = Grid(self.experiment, self.state.shape[0], self.device)

    def __init_state(self):
        from simulate.Grid import Grid
        self.save_id = 0
        self.t = 0.0
        self.grid = Grid(self.experiment, self.run_config.spatial_resolution, self.device)

        self.state = self.experiment.ic(self.grid.x, self.grid.y)

        # Apply boundary condition
        self.state[0, :] = 0
        self.state[:, 0] = 0
        self.state[-1, :] = 0
        self.state[:, -1] = 0

    def get_iterator(self):
        t_iterator = self.__get_t_linspace(self.run_config.time_resolution)
        t_iterator = t_iterator[t_iterator > self.t]

        if self.run_config.silent:
            return t_iterator

        from rich.progress import track
        return track(t_iterator)

    def get_saver(self):
        from simulate.Saver import Saver
        return Saver(self.run_config.output, self.__get_t_linspace(100), self.save_id)

    def get_initial_state(self):
        return self.state.to(self.device)

    def get_grid(self):
        return self.grid

    def get_dt(self):
        t_start, t_end = self.experiment.timespaceDomain.timeDomain
        return (t_end - t_start) / (self.run_config.time_resolution - 1)

    def __get_t_linspace(self, steps_num):
        t_start, t_end = self.experiment.timespaceDomain.timeDomain
        return torch.linspace(t_start, t_end, steps_num)
