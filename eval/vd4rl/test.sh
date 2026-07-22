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
export WANDB_API_KEY=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
ps -eo nlwp | tail -n +2 | awk '{ num_threads += $1 } END { print num_threads }'
#rm -r logdir/$ALGO/$TASK/$EXPECTILE/$SEED/
echo "CUDA_VISIBLE_DEVICES=$GPU WANDB__SERVICE_WAIT=3000 python dreamerv3/main.py --logdir logdir/$ALGO/$TASK/$EXPECTILE/$SEED --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --expectile $EXPECTILE"
XLA_PYTHON_CLIENT_MEM_FRACTION=.90 CUDA_VISIBLE_DEVICES=$GPU WANDB__SERVICE_WAIT=3000 python3 dreamerv3/main.py --logdir logdir/$ALGO/$TASK/$EXPECTILE/$SEED --configs $SUITE --task $TASK --algo $ALGO --seed $SEED --expectile $EXPECTILE --run.model_checkpoint logdir/model/$TASK/0/${SEED}/checkpoint.ckpt

#--run.model_checkpoint logdir/model/$TASK/$SEED/checkpoint.ckpt 
