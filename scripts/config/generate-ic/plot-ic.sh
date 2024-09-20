#!/bin/bash

if [ -z ${IC+x} ]; then
    echo "Variable IC unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..
echo Running from $(pwd)

python ./src/plot.py \
    -o ./tmp/generate-ic${IC}/base-ic \
    --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py \
    --title "Quadratic initial condition" \
    --title "for generating initial state ${IC}" \
    ic --max-u 1.0 --background-diffusion
python ./src/plot.py \
    -o ./tmp/generate-ic${IC}/treatment \
    --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py \
    --title "Treatment" \
    treat
python ./src/plot.py \
    -o ./tmp/generate-ic${IC}/diffusion \
    --experiment $SCRIPT_DIR/../generate-ic${IC}/Experiment.py \
    --title "Diffusion" \
    D
