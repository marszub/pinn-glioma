#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

if [ -f ./tmp/experiment1/pinn/training_state.pt ]; then
    python ./src/train.py -o ./tmp/experiment1/pinn/ --experiment=$SCRIPT_DIR/Experiment.py \
        -e 1_000_000 -d ./tmp/experiment1/train-data/ -v ./tmp/experiment1/valid-data/ \
        -l ./tmp/experiment1/pinn/training_state.pt
else
    python ./src/train.py -o ./tmp/experiment1/pinn/ --experiment=$SCRIPT_DIR/Experiment.py \
        -e 1_000_000 -d ./tmp/experiment1/train-data/ -v ./tmp/experiment1/valid-data/
fi
