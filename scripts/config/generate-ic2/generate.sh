#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
python ./src/simulate.py -o ./tmp/generate-ic2/ -r 4_000 -t 1_000_000 --experiment $SCRIPT_DIR/Experiment.py
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

python ./scripts/sim_state_to_ic.py ./tmp/generate-ic2/sim_state_99.pt ./tmp/generate-ic2/ic2.pt

${SCRIPT_DIR}/animate.sh
