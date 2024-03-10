#!/bin/bash
src/plot.py diffusion --input tmp/model_best.pt --fileName diffusion
src/plot.py animation --input tmp/model_best.pt --maxU 0.4 --backgroundDiffusion --fileName _animaiton
src/plot.py totalLoss --input tmp/loss_over_time.txt --fileName loss
