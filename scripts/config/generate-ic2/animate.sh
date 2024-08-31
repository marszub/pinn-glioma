#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

python ./src/plot.py -o ./tmp/generate-ic2/anim-sim --experiment $SCRIPT_DIR/Experiment.py \
--title "Generating initial condition 1" animation --max-u 1.0 ./tmp/generate-ic2
