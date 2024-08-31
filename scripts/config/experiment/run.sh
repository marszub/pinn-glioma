#!/bin/bash

if [ -z ${EXPERIMENT+x} ]; then
    echo "Variable EXPERIMENT unset. Set it to existing experiment number"
    exit
fi
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

mkdir -p ./tmp/experiment${EXPERIMENT}/
(time ${SCRIPT_DIR}/../experiment${EXPERIMENT}/simulate.sh) \
    &> ./tmp/experiment${EXPERIMENT}/simulation-log.txt
(time $SCRIPT_DIR/../experiment${EXPERIMENT}/train.sh) \
    &> ./tmp/experiment${EXPERIMENT}/simulation-log.txt
