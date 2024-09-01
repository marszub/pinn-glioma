#!/bin/python

# fmt: off
import sys
import os
sys.path.append(os.path.abspath('./src/'))

from pinn.sample.interior.DataFocusedRandom import DataFocusedRandom
from model.TimespaceDomain import TimespaceDomain
import matplotlib.pyplot as plt
# fmt: on

timespace = TimespaceDomain(
    spaceDomains=[(0.0, 50.0), (0.0, 80.0)],
    timeDomain=(0.0, 100.0),
)
space = DataFocusedRandom(timespace, 1000000, [9.10, 29.30, 49.49], rate=0.2)
_, _, t = space.get_points()
t = t.detach().numpy()

plt.hist(t, bins=100)
plt.xlabel("time [days]")
plt.ylabel("number of points")
plt.savefig("./tmp/sample_space.png")
