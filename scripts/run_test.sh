#!/bin/bash

if [[ -z "$1" ]]; then
    echo "Requires test name as an argument."
    exit
fi

docker run --gpus all --ipc=host --rm -it \
--mount type=bind,source="$(pwd)",target=/workspace \
--workdir /workspace \
nvcr.io/nvidia/pytorch:23.04-py3 \
bash -c "python -m pip install nibabel rich ; chmod u+x ./src/test/test_${1}.sh ; mkdir ./tmp/test/\$(hostname) ; WORKDIR=/workspace/tmp/test/\$(hostname) ./src/test/test_${1}.sh ; exec /bin/bash"
