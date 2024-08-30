#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

/workspace/src/plot.py -o ./tmp/experiment1/constraints-plot/ic \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Initial condition for Experiment 1" \
    ic --background-diffusion
/workspace/src/plot.py -o ./tmp/experiment1/constraints-plot/treatment \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Treatment in Experiment 1" treat
/workspace/src/plot.py -o ./tmp/experiment1/constraints-plot/diffusion \
    --experiment=$SCRIPT_DIR/Experiment.py --title "Diffusion in Experiment 1" D
