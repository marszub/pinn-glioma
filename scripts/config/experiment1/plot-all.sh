#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/../../..

$SCRIPT_DIR/plot-ic.sh
$SCRIPT_DIR/plot-sim.sh
$SCRIPT_DIR/plot-final.sh
