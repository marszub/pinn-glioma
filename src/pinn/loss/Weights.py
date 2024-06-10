from typing import NamedTuple

class Weights(NamedTuple):
    residual: float
    initial: float
    boundary: float
    data: float

    def normalized(self):
        total = self.residual + self.initial + self.boundary + self.data
        return Weights(
            residual=self.residual / total,
            initial=self.initial / total,
            boundary=self.boundary / total,
            data=self.data / total,
        )
