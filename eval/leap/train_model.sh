#!/bin/bash -l
# Train DreamerV3 world model only (--algo model) on converted Leap episodes.
#
# 1. Collect:  python data_collection/data_collection.py --config data_collection/config/squeeze.yaml
#              (from sparsh-skin-sim; increase collection.episodes for a real run)
# 2. Convert:  python data/leap/convert_episodes.py --input <collection_dir> --output data/leap/squeeze
# 3. Train:    ./eval/leap/train_model.sh [GPU] [SEED] [TASK]
#
# W&B (on by default): set WANDB_API_KEY in /workspace/LEQ/docker/.env or the
# environment. Override with WANDB_PROJECT / WANDB_NAME, or WANDB_ENABLE=False.
set -euo pipefail

GPU=${1:-0}
SEED=${2:-0}
TASK=${3:-leap_squeeze}
SUITE=leap
ALGO=model
WANDB_ENABLE=${WANDB_ENABLE:-True}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3_LEQ}
WANDB_NAME=${WANDB_NAME:-${ALGO}_${TASK}_${SEED}}

cd "$(dirname "$0")/../.."

if [ -f /workspace/LEQ/docker/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LEQ/docker/.env
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
    --logdir "logdir/$ALGO/$TASK/0/$SEED" \
    --configs "$SUITE" \
    --task "$TASK" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --wandb.enable "$WANDB_ENABLE" \
    --wandb.project "$WANDB_PROJECT" \
    --wandb.name "$WANDB_NAME"
