#!/bin/bash
src/plot.py diffusion --input $1/model_best.pt --output $1 --fileName diffusion
src/plot.py totalLoss --input $1/loss_over_time.txt --output $1 --fileName loss
src/plot.py sizeOverTime --input $1/model_best.pt --output $1 --fileName size
src/plot.py treatment --input $1/model_best.pt --output $1 --fileName treatment
src/plot.py animation --input $1/model_best.pt --output $1 --maxU 0.5 --backgroundDiffusion --fileName _animaiton
