#!/bin/bash -l
# Train LEQ (--algo leq) on converted Leap episodes, freezing a trained world model.
#
# Prerequisite: ./eval/leap/train_model.sh [GPU] [SEED] [TASK]
#
# Usage: ./eval/leap/train_leq.sh [GPU] [SEED] [TASK] [EXPECTILE]
#   defaults: GPU=0  SEED=0  TASK=leap_squeeze  EXPECTILE=0.1
#
# Loads logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt unless MODEL_CHECKPOINT is set.
# Writes logdir/leq/<TASK>/<EXPECTILE>/<SEED>/
#
# W&B (on by default): set WANDB_API_KEY in /workspace/LEQ/docker/.env or the
# environment. Override with WANDB_PROJECT / WANDB_NAME, or WANDB_ENABLE=False.
set -euo pipefail

GPU=${1:-0}
SEED=${2:-0}
TASK=${3:-leap_squeeze}
EXPECTILE=${4:-0.1}
SUITE=leap
ALGO=leq
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-logdir/model/${TASK}/0/${SEED}/checkpoint.ckpt}
WANDB_ENABLE=${WANDB_ENABLE:-True}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3_LEQ}
WANDB_NAME=${WANDB_NAME:-${ALGO}_${TASK}_${EXPECTILE}_${SEED}}

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

if [ ! -f "$MODEL_CHECKPOINT" ]; then
  echo "Missing world-model checkpoint: $MODEL_CHECKPOINT" >&2
  echo "Train it first, e.g. ./eval/leap/train_model.sh $GPU $SEED $TASK" >&2
  echo "Or set MODEL_CHECKPOINT to an existing checkpoint.ckpt." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  read -r GPU_USED GPU_TOTAL <<< "$(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits -i "$GPU" | head -1 | tr ',' ' ')"
  if [ -n "${GPU_USED:-}" ] && [ "${GPU_TOTAL:-0}" -gt 0 ]; then
    GPU_PCT=$((GPU_USED * 100 / GPU_TOTAL))
    if [ "$GPU_PCT" -gt 20 ]; then
      echo "Warning: GPU $GPU is ${GPU_PCT}% full (${GPU_USED}/${GPU_TOTAL} MiB)." >&2
      echo "         Another job may be running. Use a free GPU, e.g. ./train_leq.sh 1 $SEED" >&2
    fi
  fi
fi

echo "CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$XLA_MEM_FRACTION python3 dreamerv3/main.py --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --expectile $EXPECTILE --run.model_checkpoint $MODEL_CHECKPOINT --wandb.enable $WANDB_ENABLE --wandb.project $WANDB_PROJECT --wandb.name $WANDB_NAME"
CUDA_VISIBLE_DEVICES=$GPU \
  python3 dreamerv3/main.py \
    --logdir "logdir/$ALGO/$TASK/$EXPECTILE/$SEED" \
    --configs "$SUITE" \
    --task "$TASK" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --expectile "$EXPECTILE" \
    --run.model_checkpoint "$MODEL_CHECKPOINT" \
    --loss_scales.reward 1.0 \
    --loss_scales.actor 1.0 \
    --loss_scales.critic 1.0 \
    --loss_scales.value 1.0 \
    --wandb.enable "$WANDB_ENABLE" \
    --wandb.project "$WANDB_PROJECT" \
    --wandb.name "$WANDB_NAME"
