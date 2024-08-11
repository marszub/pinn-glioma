#!/bin/bash

err=0
trap 'err=1' ERR
echo Test start

/workspace/src/plot.py -o ${WORKDIR}/constraints-plot/ic \
--title "Initial condition" ic --background-diffusion
/workspace/src/plot.py -o ${WORKDIR}/constraints-plot/treatment \
--title "Treatment" treat
/workspace/src/plot.py -o ${WORKDIR}/constraints-plot/diffusion \
--title "Diffusion" D

/workspace/src/simulate.py -o ${WORKDIR}/simulation/
/workspace/src/plot.py -o ${WORKDIR}/simulation-plots/anim-sim \
--title "Simulation animation" animation --max-u 0.5 ${WORKDIR}/simulation
/workspace/src/plot.py -o ${WORKDIR}/simulation-plots/sot-sim \
--title "Simulation sot" sot ${WORKDIR}/simulation

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
mv ${WORKDIR}/simulation/sim_state_99.pt ${WORKDIR}/valid-data/

/workspace/src/train.py -o ${WORKDIR}/pinn/ -e 20000 \
-d ${WORKDIR}/train-data/ -v ${WORKDIR}/valid-data/
/workspace/src/plot.py -o ${WORKDIR}/pinn-plots/anim-pinn \
--title "PINN animation" animation --max-u 0.5 ${WORKDIR}/pinn/model_best.pt
/workspace/src/plot.py -o ${WORKDIR}/pinn-plots/sot-pinn \
--title "Simulation sot" sot ${WORKDIR}/pinn/model_best.pt
/workspace/src/plot.py -o ${WORKDIR}/pinn-plots/loss \
--title "PINN loss" loss ${WORKDIR}/pinn/loss_over_time.txt
/workspace/src/plot.py -o ${WORKDIR}/pinn-plots/total-loss \
--title "PINN total loss" total_loss ${WORKDIR}/pinn/loss_over_time.txt
/workspace/src/plot.py -o ${WORKDIR}/pinn-plots/difference \
--title "PINN and simualtion difference" difference \
${WORKDIR}/pinn/model_best.pt ${WORKDIR}/simulation

echo Test end
if (( err )); then
    echo Some steps failed. See errors above and generated results in ${WORKDIR}.
else
    echo Test finished successfully. Check generated results in ${WORKDIR}.
fi
