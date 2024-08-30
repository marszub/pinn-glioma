#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
./src/simulate.py -o ./tmp/generate-ic1/ -r 4_000 -t 1_000_000 --experiment $SCRIPT_DIR/Experiment.py
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

./scripts/sim_state_to_ic.py ./tmp/generate-ic1/sim_state_99.pt ./tmp/generate-ic1/ic1.pt

/workspace/src/plot.py -o ./tmp/generate-ic1/anim-sim --experiment $SCRIPT_DIR/Experiment.py \
--title "Generating initial condition 1" animation --max-u 1.0 ./tmp/generate-ic1
