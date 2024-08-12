#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../..
echo Running from $(pwd)

if [ -f ./tmp/pinn/training_state.pt ]; then
    ./src/train.py -i -o ./tmp/pinn/ -e 1000000 -d ./tmp/train-data/ -v ./tmp/valid-data/ \
        -l ./tmp/pinn/training_state.pt
else
    ./src/train.py -i -o ./tmp/pinn/ -e 1000000 -d ./tmp/train-data/ -v ./tmp/valid-data/
fi


/workspace/src/plot.py -o ./tmp/pinn-plots/anim-pinn \
--title "PINN animation" animation --max-u 0.5 ./tmp/pinn/model_best.pt
/workspace/src/plot.py -o ./tmp/pinn-plots/sot-pinn \
--title "Simulation sot" sot ./tmp/pinn/model_best.pt
/workspace/src/plot.py -o ./tmp/pinn-plots/loss \
--title "PINN loss" loss ./tmp/pinn/loss_over_time.txt
/workspace/src/plot.py -o ./tmp/pinn-plots/total-loss \
--title "PINN total loss" total_loss ./tmp/pinn/loss_over_time.txt
/workspace/src/plot.py -o ./tmp/pinn-plots/difference \
--title "PINN and simualtion difference" difference \
./tmp/pinn/model_best.pt ./tmp/simulation
