#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/fast-check/treatment \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Treatment in Experiment ${EXPERIMENT}" treat

python ./src/simulate.py -s -o ./tmp/experiment${EXPERIMENT}/fast-check/ \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    -r 200 -t 10_000
if [ $? -ne 0 ]; then
    echo Simulation failed
    exit
fi

python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/fast-check/sot-sim \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} GT size over time" \
    sot ./tmp/experiment${EXPERIMENT}/fast-check
python ./src/plot.py \
    -o ./tmp/experiment${EXPERIMENT}/fast-check/anim-sim \
    --experiment=$SCRIPT_DIR/../experiment${EXPERIMENT}/Experiment.py \
    --title "Experiment ${EXPERIMENT} GT" \
    animation --background-diffusion --max-u 0.8 \
    ./tmp/experiment${EXPERIMENT}/fast-check
