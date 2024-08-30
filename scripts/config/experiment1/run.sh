#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

$SCRIPT_DIR/plot-ic.sh
(time $SCRIPT_DIR/simulate.sh) &> ./tmp/experiment1/simulation-log.txt
$SCRIPT_DIR/plot-sim.sh
(time $SCRIPT_DIR/train.sh) &> ./tmp/experiment1/simulation-log.txt
$SCRIPT_DIR/plot-final.sh
