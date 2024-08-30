#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

/workspace/src/plot.py -o ./tmp/generate-ic3/anim-sim --experiment $SCRIPT_DIR/Experiment.py \
--title "Generating initial condition 3" animation --max-u 1.0 ./tmp/generate-ic3
