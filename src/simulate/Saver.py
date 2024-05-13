from torch import Tensor
from torch import save
from os import makedirs


class Saver:
    def __init__(self, dir_name: str, save_times: Tensor, file_counter: int):
        makedirs(dir_name, exist_ok=True)
        self.dir_name = dir_name
        self.save_times = save_times
        self.file_counter = file_counter

    def first_save(self, state: Tensor, t: float):
        if self.file_counter == 0:
            self.__save(state, t)

    def mid_save(self, state: Tensor, t: float):
        if self.file_counter >= self.save_times.shape[0]:
            return

        if t > self.save_times[self.file_counter]:
            self.__save(state, t)

    def last_save(self, state: Tensor, t: float):
        if self.file_counter < self.save_times.shape[0]:
            self.__save(state, t)

    def __save(self, state: Tensor, t: float):
        saved_object = {
            "save_id": self.file_counter,
            "time": t,
            "state": state
        }
        save(saved_object,
             f"{self.dir_name}/sim_state_{self.file_counter}.pt")
        self.file_counter += 1
