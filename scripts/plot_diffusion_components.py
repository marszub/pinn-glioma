#!/bin/python

# fmt: off
import sys
import os
sys.path.append(os.path.abspath('./src/'))

import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from plot.Plotter import Plotter
from importlib import reload
# fmt: on

gm = nib.load("/top/bibliography/GlioblastomaDATA/P1/gm.nii").get_fdata()
wm = nib.load("/top/bibliography/GlioblastomaDATA/P1/wm.nii").get_fdata()

x = torch.stack([torch.arange(gm.shape[0]+1)
                for i in range(gm.shape[1]+1)], dim=1)
y = torch.stack([torch.arange(gm.shape[1]+1)
                for i in range(gm.shape[0]+1)], dim=0)

gm2d = torch.tensor(gm[:, :, gm.shape[2]//3*2])  # 0.013
wm2d = torch.tensor(wm[:, :, wm.shape[2]//3*2])  # 0.13
d2d = gm2d * 0.013 + wm2d * 0.13


plotter = Plotter(cmap="bone")

plotter.plot(gm2d, x, y, "")
plt.savefig("tmp/gm.png",
            pad_inches=0,
            bbox_inches='tight',
            )

plotter.plot(wm2d, x, y, "")
plt.savefig("tmp/wm.png",
            pad_inches=0,
            bbox_inches='tight',
            )

fig = plotter.plot(d2d, x, y, "")
fig.axes[0].add_patch(
    Rectangle((19, 99), 50, 80, edgecolor='red', facecolor='none', lw=2))
plt.savefig("tmp/D.png",
            pad_inches=0,
            bbox_inches='tight',
            )
