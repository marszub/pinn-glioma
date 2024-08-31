#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
python ./src/simulate.py -s -o ./tmp/experiment${EXPERIMENT}/simulation/ \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    -r 4_000 -t 1_000_000
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

mkdir ./tmp/experiment${EXPERIMENT}/train-data
mkdir ./tmp/experiment${EXPERIMENT}/valid-data
valid_start=50
for ((i=9; i<valid_start ; i+=20)); do
    cp ./tmp/experiment${EXPERIMENT}/simulation/sim_state_${i}.pt \
        ./tmp/experiment${EXPERIMENT}/train-data
done
for ((i=$valid_start; i<100 ; i+=10)); do
    cp ./tmp/experiment${EXPERIMENT}/simulation/sim_state_${i}.pt \
        ./tmp/experiment${EXPERIMENT}/valid-data
done
