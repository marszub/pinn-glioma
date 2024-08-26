#!/bin/bash

err=0
trap 'err=1' ERR
echo Test start

/workspace/src/simulate.py -o ${WORKDIR}/simulation/

mkdir ${WORKDIR}/train-data
mkdir ${WORKDIR}/valid-data
valid_start=50
for ((i=10; i<100 ; i+=10)); do
    if (( i < valid_start )); then
        cp ${WORKDIR}/simulation/sim_state_${i}.pt ${WORKDIR}/train-data
    else
        cp ${WORKDIR}/simulation/sim_state_${i}.pt ${WORKDIR}/valid-data
    fi
done
cp ${WORKDIR}/simulation/sim_state_99.pt ${WORKDIR}/valid-data/

/workspace/src/train.py -o ${WORKDIR}/pinn/ -e 100 --neurons=20 --layers=2 \
-d ${WORKDIR}/train-data/ -v ${WORKDIR}/valid-data/

/workspace/src/train.py -l ${WORKDIR}/pinn/training_state.pt -o ${WORKDIR}/pinn/ -e 1000 --neurons=20 --layers=2 \
-d ${WORKDIR}/train-data/ -v ${WORKDIR}/valid-data/

echo Test end
if (( err )); then
    echo Some steps failed. See errors above and generated results in ${WORKDIR}.
else
    echo Test finished successfully. Check generated results in ${WORKDIR}.
fi
