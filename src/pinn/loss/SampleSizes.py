from typing import NamedTuple


class SampleSizes(NamedTuple):
    boundary: int
    initial: int
    data: int
    interior: int
    validation: int

    @staticmethod
    def scaled(scale: int):
        return SampleSizes(
            boundary=scale//4,
            initial=scale**2,
            data=scale**2,
            interior=scale**3,
            validation=scale**2
        )
