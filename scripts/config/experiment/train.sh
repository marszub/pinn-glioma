#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

if [ -f ./tmp/experiment${EXPERIMENT}/pinn/training_state.pt ]; then
    python ./src/train.py \
        -o ./tmp/experiment${EXPERIMENT}/pinn/ \
        --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
        -e 1_000_000 \
        -d ./tmp/experiment${EXPERIMENT}/train-data/ \
        -v ./tmp/experiment${EXPERIMENT}/valid-data/ \
        -l ./tmp/experiment${EXPERIMENT}/pinn/training_state.pt
else
    python ./src/train.py \
        -o ./tmp/experiment${EXPERIMENT}/pinn/ \
        --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
        -e 1_000_000 \
        -d ./tmp/experiment${EXPERIMENT}/train-data/ \
        -v ./tmp/experiment${EXPERIMENT}/valid-data/
fi
