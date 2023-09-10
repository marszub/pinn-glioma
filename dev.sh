#!/bin/bash
docker run --gpus all --ipc=host -itd --rm \
--mount type=bind,source="$(pwd)",target=/workspace \
--workdir /workspace \
--name torch_cuda \
nvcr.io/nvidia/pytorch:23.04-py3