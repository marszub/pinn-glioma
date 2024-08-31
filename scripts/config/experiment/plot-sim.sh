#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/simulation-plots/anim-sim \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} GT" \
    animation --background-diffusion --max-u 0.8 \
    ./tmp/experiment${EXPERIMENT}/simulation
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/simulation-plots/sot-sim \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} GT size over time" \
    sot ./tmp/experiment${EXPERIMENT}/simulation
