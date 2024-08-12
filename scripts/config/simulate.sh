#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../..
echo Running from $(pwd)
rm -fr ./tmp/simulation ./tmp/train-data ./tmp/valid-data > /dev/null
./src/simulate.py -o ./tmp/simulation/ -r 5000 -t 100000
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

mkdir ./tmp/train-data
mkdir ./tmp/valid-data
valid_start=50
for ((i=9; i<valid_start ; i+=20)); do
    cp ./tmp/simulation/sim_state_${i}.pt ./tmp/train-data
done
for ((i=$valid_start; i<100 ; i+=10)); do
    cp ./tmp/simulation/sim_state_${i}.pt ./tmp/valid-data
done

/workspace/src/plot.py -o ./tmp/simulation-plots/anim-sim \
--title "Simulation animation" animation --max-u 0.5 ./tmp/simulation
/workspace/src/plot.py -o ./tmp/simulation-plots/sot-sim \
--title "Simulation sot" sot ./tmp/simulation
