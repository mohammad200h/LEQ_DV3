#!/bin/bash -l
# Launch Gradio visualizer for a trained mechanical pen world model.
set -euo pipefail

LOGDIR=${1:-logdir/model/leap_mechanical_pen/0/0}
DATA=${2:-data/leap/mechanical_pen}
PORT=${3:-7860}

cd "$(dirname "$0")/../.."

if [ -f /workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env
  set +a
fi

python3 -m pip install -q gradio matplotlib

echo "Logdir: $LOGDIR"
echo "Data:   $DATA"
echo "URL:    http://localhost:$PORT"

python3 eval/mechanical_pen/visualize_model.py \
  --logdir "$LOGDIR" \
  --data "$DATA" \
  --port "$PORT"
