#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

/workspace/src/plot.py -o ./tmp/generate-ic1/base-ic --experiment $SCRIPT_DIR/Experiment.py \
--title "Base initial condition" ic --background-diffusion
/workspace/src/plot.py -o ./tmp/generate-ic1/treatment --experiment $SCRIPT_DIR/Experiment.py \
--title "Treatment" treat
/workspace/src/plot.py -o ./tmp/generate-ic1/diffusion --experiment $SCRIPT_DIR/Experiment.py \
--title "Diffusion" D
