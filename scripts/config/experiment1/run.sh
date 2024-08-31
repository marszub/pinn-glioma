#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

mkdir -p ./tmp/experiment1/
(time $SCRIPT_DIR/simulate.sh) &> ./tmp/experiment1/simulation-log.txt
(time $SCRIPT_DIR/train.sh) &> ./tmp/experiment1/simulation-log.txt
