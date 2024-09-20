#!/bin/bash

if [ -z ${IC+x} ]; then
    echo "Variable IC unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
python ./src/simulate.py -o ./tmp/generate-ic${IC}/ -r 4_000 -t 1_000_000 --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

python ./scripts/sim_state_to_ic.py \
    ./tmp/generate-ic${IC}/sim_state_99.pt \
    ./tmp/generate-ic${IC}/ic${IC}.pt

python ./src/plot.py \
    -o ./tmp/generate-ic${IC}/anim-sim \
    --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py \
    --title "Generated initial state ${IC}" \
    animation --max-u 1.0 \
    ./tmp/generate-ic${IC}
