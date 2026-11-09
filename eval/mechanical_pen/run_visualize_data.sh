#!/bin/bash -l
# Launch the Gradio data explorer for the mechanical pen demos (no model or GPU needed).
#
# Usage: ./eval/mechanical_pen/run_visualize_data.sh [DATA] [PORT] [LOGDIR]
#   defaults: data/leap/mechanical_pen  7861  logdir/model/leap_mechanical_pen/0/0
#
# LOGDIR is only used to read <LOGDIR>/val_files.txt (the held-out episodes). Without
# it, the explorer recomputes the same split main.py uses (val_frac 0.1, seed 0).
# The course in eval/mechanical_pen/course/ explains every tab.
set -euo pipefail

DATA=${1:-data/leap/mechanical_pen}
PORT=${2:-7861}
LOGDIR=${3:-logdir/model/leap_mechanical_pen/0/0}

cd "$(dirname "$0")/../.."

python3 -m pip install -q gradio matplotlib scikit-learn

echo "Data:   $DATA"
echo "Logdir: $LOGDIR (for val_files.txt only)"
echo "URL:    http://localhost:$PORT"

# The explorer only reads .npz files: keep JAX (if imported anywhere) off the GPU.
CUDA_VISIBLE_DEVICES="" python3 eval/mechanical_pen/explore_data.py \
  --data "$DATA" \
  --logdir "$LOGDIR" \
  --port "$PORT"
