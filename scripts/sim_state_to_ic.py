#!/bin/python

import sys
import torch

assert len(sys.argv) >= 3
loaded_state = torch.load(sys.argv[1])
torch.save(loaded_state["state"], sys.argv[2])
