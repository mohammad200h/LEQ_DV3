#!/bin/bash -l
# Control-rate sweep: for each rate, build the data, train a world model, train DITTO on it,
# roll the policy out inside that world model, then compare the rates.
#
# Usage: ./eval/mechanical_pen/rate_sweep.sh
#
# Settings (environment variables):
#   RATES="10 20 50 80 100"  Rates in Hz, whole numbers up to 100. Rates that divide 100
#                            keep every step the same length. Others alternate between two
#                            step lengths (80 Hz: 1 and 2 source steps, 30 Hz: 3 and 4).
#   GPUS="1"                 GPUs to use, e.g. "0 1". Rates are dealt out round-robin and
#                            each GPU trains one run at a time.
#   STEPS=50000              Training steps for both the world model and DITTO.
#   SEED=0
#   EPISODES=3               Held-out successes to roll out at each rate.
#   START_SECONDS=0.8        Real time observed before imagining, the same at every rate.
#   IMAG_LENGTH=15           DITTO horizon in steps, so 1.5 s at 10 Hz, 0.15 s at 100 Hz.
#   SWEEP_DIR=logdir/rate_sweep
#   WANDB_ENABLE=True        Runs are named rate_sweep_<hz>hz_{model,ditto}.
#
# Time: about 40 minutes per rate on an RTX 4090 at 50k steps, so about 3.5 hours for
# the 5 default rates on one GPU.
#
# Every stage is skipped once it has finished, so rerunning the script resumes the sweep.
# A stage that fails is logged, and that rate is skipped; the other rates carry on.
# Delete <SWEEP_DIR>/<hz>hz/ to redo a rate.
#
# Output in <SWEEP_DIR>/:
#   sweep.log                   progress
#   <hz>hz/model/, ditto/       the runs, with model.log and ditto.log
#   <hz>hz/rollout/<episode>/   rollout_leq_wm.py output
#   summary.md, summary.csv, summary.png
set -euo pipefail

RATES=${RATES:-"10 20 50 80 100"}
GPUS=${GPUS:-1}
STEPS=${STEPS:-50000}
SEED=${SEED:-0}
EPISODES=${EPISODES:-3}
START_SECONDS=${START_SECONDS:-0.8}
export IMAG_LENGTH=${IMAG_LENGTH:-15}
SWEEP_DIR=${SWEEP_DIR:-logdir/rate_sweep}
export WANDB_ENABLE=${WANDB_ENABLE:-True}
SOURCE_DIR=data/leap/mechanical_pen
SOURCE_HZ=100

cd "$(dirname "$0")/../.."
mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"

log() {
  echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

for hz in $RATES; do
  if ! [[ "$hz" =~ ^[0-9]+$ ]] || [ "$hz" -lt 1 ] || [ "$hz" -gt "$SOURCE_HZ" ]; then
    echo "Rate $hz is not a whole number from 1 to $SOURCE_HZ." >&2
    exit 1
  fi
done

task_for() {
  if [ "$1" -eq "$SOURCE_HZ" ]; then
    echo leap_mechanical_pen
  else
    echo "leap_mechanical_pen_${1}hz"
  fi
}

data_rate() {
  python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["freq"])' \
    "$1/manifest.json" 2>/dev/null || true
}

# Data first, one rate at a time: it is quick and uses the CPU.
log "Sweep: rates [$RATES], GPUs [$GPUS], $STEPS steps, H=$IMAG_LENGTH, $EPISODES rollouts"
for hz in $RATES; do
  [ "$hz" -eq "$SOURCE_HZ" ] && continue
  out="data/leap/mechanical_pen_${hz}hz"
  have=$(data_rate "$out")
  if [ -n "$have" ] && python3 -c "import sys; sys.exit(abs(float('$have') - $hz) > 1e-6)"; then
    log "${hz} Hz: data already in $out"
    continue
  fi
  log "${hz} Hz: building $out"
  python3 data/leap/resample_chunks.py --hz "$hz" --input "$SOURCE_DIR" --output "$out" \
    --overwrite >> "$SWEEP_DIR/data.log" 2>&1
done

run_rate() {
  local gpu=$1 hz=$2
  local task dir start
  task=$(task_for "$hz")
  dir="$SWEEP_DIR/${hz}hz"
  mkdir -p "$dir"
  # The cosine schedule must anneal past the 1000-step warmup.
  local schedule=(--run.steps "$STEPS" --opt.anneal "$((STEPS > 1001 ? STEPS : 1001))")

  if [ ! -f "$dir/model/done" ]; then
    log "${hz} Hz: training world model on GPU $gpu"
    if ! LOGDIR="$dir/model" WANDB_NAME="rate_sweep_${hz}hz_model" \
        ./eval/mechanical_pen/train_model.sh "$gpu" "$SEED" "$task" "${schedule[@]}" \
        > "$dir/model.log" 2>&1; then
      log "${hz} Hz: world model FAILED, see $dir/model.log"
      return
    fi
    touch "$dir/model/done"
  fi

  if [ ! -f "$dir/ditto/done" ]; then
    log "${hz} Hz: training DITTO on GPU $gpu"
    if ! LOGDIR="$dir/ditto" MODEL_CHECKPOINT="$dir/model/checkpoint.ckpt" \
        WANDB_NAME="rate_sweep_${hz}hz_ditto" \
        ./eval/mechanical_pen/train_ditto.sh "$gpu" "$SEED" "$task" "${schedule[@]}" \
        > "$dir/ditto.log" 2>&1; then
      log "${hz} Hz: DITTO FAILED, see $dir/ditto.log"
      return
    fi
    touch "$dir/ditto/done"
  fi

  if [ ! -f "$dir/rollout/done" ]; then
    start=$(python3 -c "print(max(1, round($START_SECONDS * $hz)))")
    log "${hz} Hz: rolling out $EPISODES episodes from step $start on GPU $gpu"
    if ! ./eval/leap/rollout_leq_wm.sh "$gpu" "$dir/ditto" --num-episodes "$EPISODES" \
        --start "$start" --out "$dir/rollout" > "$dir/rollout.log" 2>&1; then
      log "${hz} Hz: rollout FAILED, see $dir/rollout.log"
      return
    fi
    touch "$dir/rollout/done"
  fi
  log "${hz} Hz: done"
}

read -r -a gpu_list <<< "$GPUS"
read -r -a rate_list <<< "$RATES"
pids=()
for g in "${!gpu_list[@]}"; do
  queue=()
  for i in "${!rate_list[@]}"; do
    if [ $((i % ${#gpu_list[@]})) -eq "$g" ]; then
      queue+=("${rate_list[$i]}")
    fi
  done
  [ ${#queue[@]} -eq 0 ] && continue
  (
    for hz in "${queue[@]}"; do
      run_rate "${gpu_list[$g]}" "$hz"
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "Summarising"
python3 eval/mechanical_pen/rate_sweep_summary.py --sweep "$SWEEP_DIR" --rates $RATES | tee -a "$LOG"
