#!/bin/bash -l
# Run a trained policy inside its world model and decode the result. No sim or hand needed.
#
# Usage: ./eval/leap/rollout_leq_wm.sh [GPU] [LOGDIR] [extra rollout_leq_wm.py flags...]
#   defaults: GPU=0  LOGDIR=logdir/ditto/leap_mechanical_pen/0
#
# By default this uses the first successful held-out episode, observes 8 real steps, then
# imagines to the end. Results go to <LOGDIR>/rollout_wm/<episode>/: rollout.gif, summary.png,
# rollout.npz and metrics.json.
#
# Examples:
#   ./eval/leap/rollout_leq_wm.sh 1
#   ./eval/leap/rollout_leq_wm.sh 1 logdir/ditto/leap_mechanical_pen/0 --num-episodes 5
#   ./eval/leap/rollout_leq_wm.sh 1 logdir/ditto/leap_mechanical_pen/0 --start 500 --steps 200
#   ./eval/leap/rollout_leq_wm.sh 1 <another run folder> --split all
set -euo pipefail

GPU=${1:-0}
LOGDIR=${2:-logdir/ditto/leap_mechanical_pen/0}
EXTRA_FLAGS=("${@:3}")

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}"

set -x
CUDA_VISIBLE_DEVICES="$GPU" \
  python3 eval/leap/rollout_leq_wm.py \
    --logdir "$LOGDIR" \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
