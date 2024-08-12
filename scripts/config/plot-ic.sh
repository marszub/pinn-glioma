#!/bin/bash

/workspace/src/plot.py -o ./tmp/constraints-plot/ic \
--title "Initial condition" ic --background-diffusion
/workspace/src/plot.py -o ./tmp/constraints-plot/treatment \
--title "Treatment" treat
/workspace/src/plot.py -o ./tmp/constraints-plot/diffusion \
--title "Diffusion" D
