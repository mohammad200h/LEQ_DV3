#!/bin/bash -l
# Launch Gradio visualizer for a trained Leap world model.
set -euo pipefail

LOGDIR=${1:-logdir/model/leap_squeeze/0/0}
DATA=${2:-data/leap/squeeze}
PORT=${3:-7860}

cd "$(dirname "$0")/../.."

if [ -f /workspace/LEQ/docker/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LEQ/docker/.env
  set +a
fi

python3 -m pip install -q gradio matplotlib

echo "Logdir: $LOGDIR"
echo "Data:   $DATA"
echo "URL:    http://localhost:$PORT"

python3 eval/leap/visualize_model.py \
  --logdir "$LOGDIR" \
  --data "$DATA" \
  --port "$PORT"
