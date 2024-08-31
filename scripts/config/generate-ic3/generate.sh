#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
python ./src/simulate.py -o ./tmp/generate-ic3/ -r 4_000 -t 1_000_000 --experiment $SCRIPT_DIR/Experiment.py
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

python ./scripts/sim_state_to_ic.py ./tmp/generate-ic3/sim_state_99.pt ./tmp/generate-ic3/ic3.pt

python ./src/plot.py -o ./tmp/generate-ic3/anim-sim --experiment $SCRIPT_DIR/Experiment.py \
--title "Generating initial condition 3" animation --max-u 1.0 ./tmp/generate-ic3
