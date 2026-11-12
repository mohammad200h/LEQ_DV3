#!/bin/bash -l
# Train DreamerV3 world model only (--algo model) on the mechanical pen demos.
#
# Data is already in DreamerV3 chunk format (no convert_episodes.py step):
#   data/leap/mechanical_pen -> ros_ws/demonstrations/compose_playbacks/DV3/100hz
#   Other rates: data/leap/mechanical_pen_new10hz, _new50hz (sparsh-skin run at that rate,
#   DV3/10hz and DV3/50hz) and mechanical_pen_<N>hz (resampled from 100 Hz).
#
# Usage: ./eval/mechanical_pen/train_model.sh [GPU] [SEED] [TASK] [extra main.py flags...]
#   Writes logdir/model/<TASK>/0/<SEED>/ unless LOGDIR is set.
#
# W&B (on by default): set WANDB_API_KEY in /workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env or the
# environment. Override with WANDB_PROJECT / WANDB_NAME, or WANDB_ENABLE=False.
set -euo pipefail

GPU=${1:-0}
SEED=${2:-0}
TASK=${3:-leap_mechanical_pen}
SUITE=mechanical_pen
ALGO=model
EXTRA_FLAGS=("${@:4}")
LOGDIR=${LOGDIR:-logdir/$ALGO/$TASK/0/$SEED}
WANDB_ENABLE=${WANDB_ENABLE:-True}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3_LEQ}
WANDB_NAME=${WANDB_NAME:-${ALGO}_${TASK}_${SEED}}

cd "$(dirname "$0")/../.."

if [ -f /workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env
  set +a
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  export WANDB_API_KEY
elif [ "$WANDB_ENABLE" = "True" ]; then
  echo "Warning: WANDB_API_KEY not set; using cached wandb login if available." >&2
fi
export WANDB__SERVICE_WAIT=3000
XLA_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"

if command -v nvidia-smi >/dev/null 2>&1; then
  read -r GPU_USED GPU_TOTAL <<< "$(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits -i "$GPU" | head -1 | tr ',' ' ')"
  if [ -n "${GPU_USED:-}" ] && [ "${GPU_TOTAL:-0}" -gt 0 ]; then
    GPU_PCT=$((GPU_USED * 100 / GPU_TOTAL))
    if [ "$GPU_PCT" -gt 20 ]; then
      echo "Warning: GPU $GPU is ${GPU_PCT}% full (${GPU_USED}/${GPU_TOTAL} MiB)." >&2
      echo "         Another job may be running. Use a free GPU, e.g. ./train_model.sh 1 0" >&2
    fi
  fi
fi

echo "CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION python3 dreamerv3/main.py --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --wandb.enable $WANDB_ENABLE --wandb.project $WANDB_PROJECT --wandb.name $WANDB_NAME"
CUDA_VISIBLE_DEVICES=$GPU \
  python3 dreamerv3/main.py \
    --logdir "$LOGDIR" \
    --configs "$SUITE" \
    --task "$TASK" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --wandb.enable "$WANDB_ENABLE" \
    --wandb.project "$WANDB_PROJECT" \
    --wandb.name "$WANDB_NAME" \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
