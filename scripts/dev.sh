#!/bin/bash
docker run --gpus all --ipc=host -itd --rm \
--mount type=bind,source="$(pwd)",target=/workspace \
--mount type=bind,source="$(pwd)/..",target=/top \
--workdir /workspace \
--name torch_cuda \
nvcr.io/nvidia/pytorch:23.04-py3 \
bash -c 'python -m pip install nibabel ; exec /bin/bash'
