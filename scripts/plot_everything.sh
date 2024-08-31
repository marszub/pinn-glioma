#!/bin/bash

# $1 - pinn dir
# $2 - simualtion dir
# $3 - output dir

python ./src/plot.py -o $3/constraints-plot/ic \
--title "Initial condition" ic --background-diffusion
python ./src/plot.py -o $3/constraints-plot/treatment \
--title "Treatment" treat
python ./src/plot.py -o $3/constraints-plot/diffusion \
--title "Diffusion" D


python ./src/plot.py -o $3/simulation-plots/anim-sim \
--title "Simulation animation" animation --max-u 0.5 $2
python ./src/plot.py -o $3/simulation-plots/sot-sim \
--title "Simulation sot" sot $2

python ./src/plot.py -o $3/pinn-plots/anim-pinn \
--title "PINN animation" animation --max-u 0.5 $1/model_best.pt
python ./src/plot.py -o $3/pinn-plots/sot-pinn \
--title "Simulation sot" sot $1/model_best.pt
python ./src/plot.py -o $3/pinn-plots/loss \
--title "PINN loss" loss $1/loss_over_time.txt
python ./src/plot.py -o $3/pinn-plots/total-loss \
--title "PINN total loss" total_loss $1/loss_over_time.txt
python ./src/plot.py -o $3/pinn-plots/difference \
--title "PINN and simualtion difference" difference \
$1/model_best.pt $2
