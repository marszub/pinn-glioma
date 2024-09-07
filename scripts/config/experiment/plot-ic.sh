#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/ic \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Initial state for Experiment ${EXPERIMENT}" \
    ic --background-diffusion
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/treatment \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Treatment in Experiment ${EXPERIMENT}" treat
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/constraints-plot/diffusion \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Diffusion in Experiment ${EXPERIMENT}" D
