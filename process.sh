#!/bin/bash

# Processes a case from data/<name>/<name>.json
# - outputs svg in out_<name>.svg (current directory)
# - displays width stats
#
# Exemple: ./process.sh test

# Generate curves (slow)
python tools/fill_2d_shape.py data/$1/$1.json

# Extracts polylines
# --split_radius controls the curvature-based spliting
# --smoothing controls the number of iterations of the smoothing post-process (more => smoother)
python tools/tolines.py data/$1/$1.json cycles tmp_$1.svg --split_radius 1.0 --smoothing 10 

# Simplify to cubic splines
# -s controls the tradeoff precision/number of control points
# --extremity_w controls the weight 'pulling' the path extremities (1.0 is same as everywhere else, 0.001 means 'keep going straight'
# --extremity_len defines the length, in number of vertices, of the open path extrimities
python polyline_to_bezier.py -i tmp_$1.svg -o out_$1.svg -s 0.01 --extremity_w 0.1 --extremity_len 6

# Show histogram
# python -c "import numpy as np, matplotlib.pyplot as plt; d=np.loadtxt('data/$1/np/$1_width.txt'); plt.hist(d, bins=100); plt.show()"
