#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/difference \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} PINN and GT difference" \
    difference \
    ./tmp/experiment${EXPERIMENT}/simulation \
    ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/anim-pinn \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} PINN prediction" \
    animation --background-diffusion --max-u 0.8 \
    ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/sot-pinn \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} PINN prediction" \
    --title "size over time" \
    sot ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/loss \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} PINN loss components" \
    loss ./tmp/experiment${EXPERIMENT}/pinn/loss_over_time.txt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/total-loss \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} PINN total loss" \
    total_loss ./tmp/experiment${EXPERIMENT}/pinn/loss_over_time.txt
