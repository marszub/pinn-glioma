#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)
./src/simulate.py -s -o ./tmp/experiment1/simulation/ --experiment=$SCRIPT_DIR/Experiment.py \
    -r 4_000 -t 1_000_000
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

mkdir ./tmp/experiment1/train-data
mkdir ./tmp/experiment1/valid-data
valid_start=50
for ((i=9; i<valid_start ; i+=20)); do
    cp ./tmp/experiment1/simulation/sim_state_${i}.pt ./tmp/experiment1/train-data
done
for ((i=$valid_start; i<100 ; i+=10)); do
    cp ./tmp/experiment1/simulation/sim_state_${i}.pt ./tmp/experiment1/valid-data
done
