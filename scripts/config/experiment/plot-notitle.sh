#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..


python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/ic \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    ic --background-diffusion
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/treatment \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    treat
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/diffusion \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    D
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/anim-pinn \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    animation --background-diffusion --max-u 0.8 \
    ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/sot-pinn \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    sot ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/sot-pinn-sim \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    sot ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt \
    --other-model ./tmp/experiment${EXPERIMENT}/simulation \
    --train-data ./tmp/experiment${EXPERIMENT}/train-data
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/loss \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    loss ./tmp/experiment${EXPERIMENT}/pinn/loss_over_time.txt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/total-loss \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    total_loss ./tmp/experiment${EXPERIMENT}/pinn/loss_over_time.txt
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/validation-loss \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    total_loss ./tmp/experiment${EXPERIMENT}/pinn/loss_over_time.txt \
    --validation
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/pinn-plots/difference \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    difference \
    ./tmp/experiment${EXPERIMENT}/simulation \
    ./tmp/experiment${EXPERIMENT}/pinn/model_best.pt \
    --train-data ./tmp/experiment${EXPERIMENT}/train-data
