#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

mkdir -p ./tmp/experiment${EXPERIMENT}/
(time ${SCRIPT_DIR}/simulate.sh) \
    &> ./tmp/experiment${EXPERIMENT}/simulation-log.txt
(time $SCRIPT_DIR/train.sh) \
    &> ./tmp/experiment${EXPERIMENT}/training-log.txt
