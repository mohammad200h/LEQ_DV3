#!/bin/bash -l
# Train the world model, then DITTO on it, on the synthetic 30 Hz data, for several seeds.
# Nothing else: no rollouts, posterior evaluation or summary.
#
# The synthetic data (ros_ws synthetic_data.py) has the recorded commands cleaned up
# (interpolated in time and smoothed) and replayed in MuJoCo, so actions and joints are on
# one exact 30 Hz grid, with the real Xela forces. It is linked in as
# data/leap/mechanical_pen_synthetic30hz (task leap_mechanical_pen_synthetic30hz).
#
# Usage: ./eval/mechanical_pen/train_synthetic.sh
#
# Settings (environment variables):
#   DATA=<compose_playbacks>/synthetic_data_30hz/DV3/30hz   DreamerV3 chunks to train on
#   SEEDS="0 1 2"
#   GPUS="0 1"           Seeds are dealt out round-robin; each GPU trains one run at a time.
#   STEPS=50000          Training steps for both the world model and DITTO.
#   WM_FLAGS=""          Extra world-model flags.
#   DITTO_FLAGS=""       Extra DITTO flags.
#   OUT_DIR=logdir/synthetic_30hz
#   WANDB_ENABLE=True    Runs are named synthetic30hz_s<seed>_{model,ditto}.
#
# Each stage is skipped once it has finished, so rerunning resumes. A stage that fails is
# logged and that seed is skipped. Delete <OUT_DIR>/seed<s>/ to redo a seed.
#
# Output in <OUT_DIR>/:
#   train.log                    progress
#   seed<s>/model/, ditto/       the runs (checkpoint.ckpt, config.yaml, ...)
#   seed<s>/model.log, ditto.log their output
set -euo pipefail

DATA=${DATA:-/workspace/LeapXELA_Hardware_ws/ros_ws/demonstrations/compose_playbacks/synthetic_data_30hz/DV3/30hz}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0 1"}
STEPS=${STEPS:-50000}
WM_FLAGS=${WM_FLAGS:-""}
DITTO_FLAGS=${DITTO_FLAGS:-""}
OUT_DIR=${OUT_DIR:-logdir/synthetic_30hz}
NAME=mechanical_pen_synthetic30hz
TASK=leap_$NAME
export WANDB_ENABLE=${WANDB_ENABLE:-True}

cd "$(dirname "$0")/../.."
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/train.log"
log() {
  echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

if [ ! -f "$DATA/manifest.json" ] || ! compgen -G "$DATA/*.npz" > /dev/null; then
  echo "No chunks or manifest.json in $DATA." >&2
  echo "Build them with: python3 -m mechanical_pen_data_collection.synthetic_data" >&2
  exit 1
fi
link="data/leap/$NAME"
if [ -e "$link" ] && [ "$(realpath "$link")" != "$(realpath "$DATA")" ]; then
  echo "$link already exists and points elsewhere ($(realpath "$link"))." >&2
  exit 1
fi
[ -e "$link" ] || ln -s "$(realpath "$DATA")" "$link"

run_one() {
  local gpu=$1 seed=$2
  local dir="$OUT_DIR/seed$seed"
  local wm_flags ditto_flags
  read -r -a wm_flags <<< "$WM_FLAGS"
  read -r -a ditto_flags <<< "$DITTO_FLAGS"
  mkdir -p "$dir"
  # The cosine schedule must anneal past the 1000-step warmup.
  local schedule=(--run.steps "$STEPS" --opt.anneal "$((STEPS > 1001 ? STEPS : 1001))")

  if [ ! -f "$dir/model/done" ]; then
    log "seed $seed: training world model on GPU $gpu ($TASK)"
    if ! LOGDIR="$dir/model" WANDB_NAME="synthetic30hz_s${seed}_model" \
        ./eval/mechanical_pen/train_model.sh "$gpu" "$seed" "$TASK" "${schedule[@]}" \
        "${wm_flags[@]+"${wm_flags[@]}"}" > "$dir/model.log" 2>&1; then
      log "seed $seed: world model FAILED, see $dir/model.log"
      return
    fi
    touch "$dir/model/done"
  fi

  if [ ! -f "$dir/ditto/done" ]; then
    log "seed $seed: training DITTO on GPU $gpu"
    if ! LOGDIR="$dir/ditto" MODEL_CHECKPOINT="$dir/model/checkpoint.ckpt" \
        WANDB_NAME="synthetic30hz_s${seed}_ditto" \
        ./eval/mechanical_pen/train_ditto.sh "$gpu" "$seed" "$TASK" "${schedule[@]}" \
        "${ditto_flags[@]+"${ditto_flags[@]}"}" > "$dir/ditto.log" 2>&1; then
      log "seed $seed: DITTO FAILED, see $dir/ditto.log"
      return
    fi
    touch "$dir/ditto/done"
  fi
  log "seed $seed: done"
}

log "Synthetic 30 Hz: $DATA, seeds [$SEEDS], GPUs [$GPUS], $STEPS steps," \
  "world-model flags [${WM_FLAGS:-none}], DITTO flags [${DITTO_FLAGS:-none}]"

read -r -a seed_list <<< "$SEEDS"
read -r -a gpu_list <<< "$GPUS"
pids=()
for g in "${!gpu_list[@]}"; do
  (
    for i in "${!seed_list[@]}"; do
      if [ $((i % ${#gpu_list[@]})) -eq "$g" ]; then
        run_one "${gpu_list[$g]}" "${seed_list[$i]}"
      fi
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done
log "All seeds finished"
