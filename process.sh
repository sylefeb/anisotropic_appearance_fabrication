#!/bin/bash

# Processes a case from data/<name>/<name>.json
# - outputs svg in out_<name>.svg (current directory)
# - displays width stats
#
# Exemple: ./process.sh test

# Generate curves (slow)
python tools/fill_2d_shape.py data/$1/$1.json

# Extracts polylines
python tools/tolines.py data/$1/$1.json cycles tmp_$1.svg --split_radius 1.0 --smoothing 10 

# Simplify to cubic splines
python polyline_to_bezier.py -i tmp_$1.svg -o out_$1.svg -s 0.01

# Show histogram
# python -c "import numpy as np, matplotlib.pyplot as plt; d=np.loadtxt('data/$1/np/$1_width.txt'); plt.hist(d, bins=100); plt.show()"
