#!/bin/bash -l

SUITE=$1
TASK=$2
ALGO=$3
EXPECTILE=$4
GPU=$5
SEED=$6
nvidia-smi
nvcc --version
hostname
df .

conda activate IQL
# Use local WANDB key (do not hardcode another user's key).
if [ -f /workspace/LEQ/docker/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LEQ/docker/.env
  set +a
fi
: "${WANDB_API_KEY:?Set WANDB_API_KEY (e.g. in /workspace/LEQ/docker/.env)}"
export WANDB_API_KEY
ps -eo nlwp | tail -n +2 | awk '{ num_threads += $1 } END { print num_threads }'
#rm -r logdir/$ALGO/$TASK/$EXPECTILE/$SEED/
echo "CUDA_VISIBLE_DEVICES=$GPU WANDB__SERVICE_WAIT=3000 python3 dreamerv3/main.py --logdir logdir/$ALGO/$TASK/$EXPECTILE/$SEED --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --expectile $EXPECTILE"
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 CUDA_VISIBLE_DEVICES=$GPU WANDB__SERVICE_WAIT=3000 python3 dreamerv3/main.py --logdir logdir/$ALGO/$TASK/$EXPECTILE/$SEED --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --expectile $EXPECTILE --run.eval_eps 1 --run.num_envs_eval 1 

#--run.model_checkpoint logdir/model/$TASK/$SEED/checkpoint.ckpt 
