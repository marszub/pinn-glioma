#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

python ./src/plot.py -o ./tmp/experiment1/pinn-plots/anim-pinn \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 PINN prediction" \
    animation --background-diffusion --max-u 0.8 ./tmp/experiment1/pinn/model_best.pt
python ./src/plot.py -o ./tmp/experiment1/pinn-plots/sot-pinn \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 PINN prediction" \
    --title "size over time" \
    sot ./tmp/experiment1/pinn/model_best.pt
python ./src/plot.py -o ./tmp/experiment1/pinn-plots/loss \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 PINN loss components" \
    loss ./tmp/experiment1/pinn/loss_over_time.txt
python ./src/plot.py -o ./tmp/experiment1/pinn-plots/total-loss \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 PINN total loss" \
    total_loss ./tmp/experiment1/pinn/loss_over_time.txt
python ./src/plot.py -o ./tmp/experiment1/pinn-plots/difference \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experimetn 1 PINN and GT difference" \
    difference ./tmp/experiment1/pinn/model_best.pt ./tmp/experiment1/simulation
