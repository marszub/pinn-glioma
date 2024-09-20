#!/bin/bash

if [ -z ${IC+x} ]; then
    echo "Variable IC unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

python ./src/plot.py \
    -o ./tmp/generate-ic${IC}/anim-sim \
    --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py \
    --title "Generated initial state 1" \
    animation --background-diffusion --max-u 1.0 \
    ./tmp/generate-ic${IC}
