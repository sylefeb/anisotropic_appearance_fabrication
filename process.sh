#!/bin/bash

# Processes a case from data/<name>/<name>.json
# - outputs svg in out_<name>.svg (current directory)
# - displays width stats
#
# Exemple: ./process.sh test

python tools/fill_2d_shape.py data/$1/$1.json
python tools/tolines.py data/$1/$1.json cycles out_$1.svg
python -c "import numpy as np, matplotlib.pyplot as plt; d=np.loadtxt('data/$1/np/$1_width.txt'); plt.hist(d, bins=100); plt.show()"
