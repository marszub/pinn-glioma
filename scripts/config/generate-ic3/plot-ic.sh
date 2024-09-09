#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

python ./src/plot.py -o ./tmp/generate-ic3/base-ic --experiment $SCRIPT_DIR/Experiment.py \
--title "Quadratic initial condition" --title "for generating initial state 3" ic --max-u 1.0 \
--background-diffusion
python ./src/plot.py -o ./tmp/generate-ic3/treatment --experiment $SCRIPT_DIR/Experiment.py \
--title "Treatment" treat
python ./src/plot.py -o ./tmp/generate-ic3/diffusion --experiment $SCRIPT_DIR/Experiment.py \
--title "Diffusion" D
