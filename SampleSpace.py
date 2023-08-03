from typing import NamedTuple
from TimespaceDomain import TimespaceDomain

class SampleSpace(NamedTuple):
    timespaceDomain: TimespaceDomain
    spaceResoultion: int
    timeResoultion: int
