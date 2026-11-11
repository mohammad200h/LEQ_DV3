#!/bin/bash -l
# Data sweep: train the world model, DITTO and rollouts on several datasets, each with
# several seeds, then compare how jerky each one is.
#
# Use it to test a change in how the data was made (for example sparsh-skin run at the
# target rate instead of resampled from 100 Hz). Every run uses the same default settings,
# so only the data differs.
#
# Usage: ./eval/mechanical_pen/run_sweep_data.sh
#
# Settings (environment variables):
#   DATASETS="label=dir ..."  Datasets to compare. dir is a folder of DreamerV3 chunks with
#                             a manifest.json (its "freq" sets the rate). A folder outside
#                             data/leap/ is linked in as data/leap/mechanical_pen_<label>.
#                             Default: the sparsh-skin-at-rate data in compose_playbacks/DV3
#                             (new10hz, new50hz, new100hz) and the resampled 50 Hz data
#                             (old50hz).
#   SEEDS="0 1 2"
#   GPUS="0 1"                Runs are dealt out round-robin; each GPU trains one at a time.
#   STEPS=50000               Training steps for both the world model and DITTO.
#   WM_FLAGS=""               Extra world-model flags, the same for every run.
#   EPISODES=3                Held-out successes to roll out per run.
#   START_SECONDS=0.8         Real time observed before imagining.
#   SWEEP_DIR=logdir/data_sweep
#   WANDB_ENABLE=True         Runs are named data_sweep_<label>_s<seed>_{model,ditto}.
#
# Time: about 40 minutes per run on an RTX 4090 at 50k steps, so about 4.5 hours for the
# default 12 runs on two GPUs.
#
# Every stage is skipped once it has finished, so rerunning resumes the sweep. A stage that
# fails is logged and that run is skipped. Delete <SWEEP_DIR>/seed<s>/<label>/ to redo one.
#
# Output in <SWEEP_DIR>/:
#   sweep.log                         progress
#   seed<s>/<label>/model, ditto      the runs, with model.log and ditto.log
#   seed<s>/<label>/rollout/<ep>/     rollout.gif, summary.png, metrics.json
#   summary.md, summary.png           mean and spread over seeds, jerkiness measures
#
# To add a reference that was trained elsewhere with the same settings, link its setting
# folder in as <SWEEP_DIR>/seed<s>/<label> and rerun the summary:
#   python3 eval/mechanical_pen/posterior_seed_summary.py --metrics jitter \
#       --sweeps <SWEEP_DIR>/seed0 <SWEEP_DIR>/seed1 <SWEEP_DIR>/seed2 \
#       --configs <labels...> --out <SWEEP_DIR>
set -euo pipefail

DV3=/workspace/LeapXELA_Hardware_ws/ros_ws/demonstrations/compose_playbacks/DV3
DATASETS=${DATASETS:-"new10hz=$DV3/10hz new50hz=$DV3/50hz new100hz=$DV3/100hz old50hz=data/leap/mechanical_pen_50hz"}
SEEDS=${SEEDS:-"0 1 2"}
GPUS=${GPUS:-"0 1"}
STEPS=${STEPS:-50000}
WM_FLAGS=${WM_FLAGS:-""}
EPISODES=${EPISODES:-3}
START_SECONDS=${START_SECONDS:-0.8}
SWEEP_DIR=${SWEEP_DIR:-logdir/data_sweep}
export WANDB_ENABLE=${WANDB_ENABLE:-True}

cd "$(dirname "$0")/../.."
mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"
log() {
  echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

# label -> task, linking outside folders into data/leap/.
declare -A TASKS
labels=()
for entry in $DATASETS; do
  label=${entry%%=*}
  dir=${entry#*=}
  if [ ! -f "$dir/manifest.json" ] || ! compgen -G "$dir/*.npz" > /dev/null; then
    echo "$label: no chunks or manifest.json in $dir" >&2
    exit 1
  fi
  real=$(realpath "$dir")
  if [ "$(dirname "$real")" = "$(realpath data/leap)" ]; then
    name=$(basename "$real")
  else
    name="mechanical_pen_$label"
    link="data/leap/$name"
    if [ -e "$link" ] && [ "$(realpath "$link")" != "$real" ]; then
      echo "$link already exists and points elsewhere; rename the label." >&2
      exit 1
    fi
    [ -e "$link" ] || ln -s "$real" "$link"
  fi
  TASKS[$label]="leap_$name"
  labels+=("$label")
done

run_one() {
  local gpu=$1 label=$2 seed=$3
  local task=${TASKS[$label]}
  local dir="$SWEEP_DIR/seed$seed/$label"
  local tag="$label seed $seed"
  local wm_flags
  read -r -a wm_flags <<< "$WM_FLAGS"
  mkdir -p "$dir"
  # The cosine schedule must anneal past the 1000-step warmup.
  local schedule=(--run.steps "$STEPS" --opt.anneal "$((STEPS > 1001 ? STEPS : 1001))")

  if [ ! -f "$dir/model/done" ]; then
    log "$tag: training world model on GPU $gpu ($task)"
    if ! LOGDIR="$dir/model" WANDB_NAME="data_sweep_${label}_s${seed}_model" \
        ./eval/mechanical_pen/train_model.sh "$gpu" "$seed" "$task" "${schedule[@]}" \
        "${wm_flags[@]+"${wm_flags[@]}"}" > "$dir/model.log" 2>&1; then
      log "$tag: world model FAILED, see $dir/model.log"
      return
    fi
    touch "$dir/model/done"
  fi

  if [ ! -f "$dir/ditto/done" ]; then
    log "$tag: training DITTO on GPU $gpu"
    if ! LOGDIR="$dir/ditto" MODEL_CHECKPOINT="$dir/model/checkpoint.ckpt" \
        WANDB_NAME="data_sweep_${label}_s${seed}_ditto" \
        ./eval/mechanical_pen/train_ditto.sh "$gpu" "$seed" "$task" "${schedule[@]}" \
        > "$dir/ditto.log" 2>&1; then
      log "$tag: DITTO FAILED, see $dir/ditto.log"
      return
    fi
    touch "$dir/ditto/done"
  fi

  if [ ! -f "$dir/rollout/done" ]; then
    log "$tag: rolling out $EPISODES episodes from ${START_SECONDS} s on GPU $gpu"
    if ! EPISODES="$EPISODES" START_SECONDS="$START_SECONDS" \
        ./eval/mechanical_pen/rollout_leq_wm.sh "$gpu" "$dir/ditto" --out "$dir/rollout" \
        > "$dir/rollout.log" 2>&1; then
      log "$tag: rollout FAILED, see $dir/rollout.log"
      return
    fi
    touch "$dir/rollout/done"
  fi
  log "$tag: done"
}

log "Data sweep: [${labels[*]}], seeds [$SEEDS], GPUs [$GPUS], $STEPS steps," \
  "world-model flags [${WM_FLAGS:-none}]"

# Seed-major order, so every dataset gets its first seed before any gets a third.
jobs=()
for seed in $SEEDS; do
  for label in "${labels[@]}"; do
    jobs+=("$label $seed")
  done
done
read -r -a gpu_list <<< "$GPUS"
pids=()
for g in "${!gpu_list[@]}"; do
  (
    for i in "${!jobs[@]}"; do
      if [ $((i % ${#gpu_list[@]})) -eq "$g" ]; then
        read -r label seed <<< "${jobs[$i]}"
        run_one "${gpu_list[$g]}" "$label" "$seed"
      fi
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "Summarising"
sweeps=()
for seed in $SEEDS; do
  sweeps+=("$SWEEP_DIR/seed$seed")
done
python3 eval/mechanical_pen/posterior_seed_summary.py --metrics jitter \
  --sweeps "${sweeps[@]}" --configs "${labels[@]}" --out "$SWEEP_DIR" | tee -a "$LOG"
