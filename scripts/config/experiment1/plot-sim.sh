#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

/workspace/src/plot.py -o ./tmp/experiment1/simulation-plots/anim-sim \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 GT" \
    animation --background-diffusion --max-u 0.8 ./tmp/experiment1/simulation
/workspace/src/plot.py -o ./tmp/experiment1/simulation-plots/sot-sim \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Experiment 1 GT size over time" \
    sot ./tmp/experiment1/simulation
