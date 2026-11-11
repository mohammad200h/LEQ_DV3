#!/bin/bash -l
# Posterior-certainty sweep: retrain the world model with settings that should make the
# posterior over z sharper, train DITTO on each, roll the policy out, then compare.
#
# Why: on held-out data the posterior keeps about 5-6 equally likely classes per z variable,
# barely fewer than the prior, so every draw of z is nearly random (see WorldModelJitter.md).
# The settings test three levers, each by itself and then together:
#
#   1. Lower rep: less pull of the posterior toward the wide prior.
#   2. Lower free: a lower KL floor, so the model can't sit at 1 nat for free.
#   3. Stronger forces decoder loss: more pressure for z to carry what the sensors show.
#      On its own this risks a posterior that is confidently wrong on new episodes
#      (held-out forces already reconstruct about 4x worse than training), so it is also
#      run with regularisation (weight decay) and with augmentation (noisy encoder input).
#      More data is the third remedy; it isn't part of this sweep.
#
#   baseline         current settings (free 1.0, rep 0.1, forces x1), for reference
#   rep0.01          1: rep 0.1 -> 0.01
#   free0.1          2: KL floor 1.0 -> 0.1 nat
#   free0            2: no KL floor
#   free0.1_rep0.01  1 + 2
#   forces3          3: forces decoder loss x3 (forces_loss_scale; positions unchanged)
#   forces3_wd       3 + weight decay 0.1 on kernels (opt.wd)
#   forces3_noise    3 + Gaussian noise of 0.1 z-score units on the encoder's forces input
#                    (forces_aug_noise); the decoder still reconstructs the clean forces
#   all              1 + 2 + 3 + weight decay + noise
#
# To add a setting, add a line to flags_for() below and name it in CONFIGS. Compare held-out
# against training columns in the summary to see whether a sharper posterior generalises.
#
# Usage: ./eval/mechanical_pen/run_sweep_posterior.sh
#
# Settings (environment variables):
#   CONFIGS="baseline rep0.01 free0.1 free0 free0.1_rep0.01 forces3 forces3_wd
#            forces3_noise all"
#   RATE=10                  Data rate in Hz. Needs data/leap/mechanical_pen_<RATE>hz, made
#                            by run_sweep_hz.sh or data/leap/resample_chunks.py.
#   GPUS="0 1"               GPUs to use. Settings are dealt out round-robin and each GPU
#                            trains one run at a time.
#   STEPS=50000              Training steps for both the world model and DITTO.
#   SEED=0
#   MODEL_ONLY=False         True trains the world models only. Entropy, KL and
#                            reconstruction need only these; DITTO and rollouts show the
#                            effect on the policy.
#   EPISODES=3               Held-out successes to roll out per setting.
#   START_SECONDS=0.8        Real time observed before imagining.
#   SWEEP_DIR=logdir/posterior_sweep_<RATE>hz
#   WANDB_ENABLE=True        Runs are named posterior_sweep_<RATE>hz_<setting>_{model,ditto}.
#
# Time: about 40 minutes per setting on an RTX 4090 at 50k steps (15 minutes with
# MODEL_ONLY=True), so about 3.5 hours for the 9 default settings on two GPUs
# (about 1.3 hours with MODEL_ONLY=True).
#
# Every stage is skipped once it has finished, so rerunning the script resumes the sweep.
# A stage that fails is logged, and that setting is skipped; the others carry on.
# Delete <SWEEP_DIR>/<setting>/ to redo a setting.
#
# Output in <SWEEP_DIR>/:
#   sweep.log                       progress
#   <setting>/model/, ditto/        the runs, with model.log and ditto.log
#   <setting>/rollout/<episode>/    rollout output (GIF, summary.png, metrics.json)
#   summary.md, summary.csv, summary.png
set -euo pipefail

CONFIGS=${CONFIGS:-"baseline rep0.01 free0.1 free0 free0.1_rep0.01 forces3 forces3_wd forces3_noise all"}
RATE=${RATE:-10}
GPUS=${GPUS:-"0 1"}
STEPS=${STEPS:-50000}
SEED=${SEED:-0}
MODEL_ONLY=${MODEL_ONLY:-False}
EPISODES=${EPISODES:-3}
START_SECONDS=${START_SECONDS:-0.8}
SWEEP_DIR=${SWEEP_DIR:-logdir/posterior_sweep_${RATE}hz}
export WANDB_ENABLE=${WANDB_ENABLE:-True}

# World-model flags for each setting.
flags_for() {
  case "$1" in
    baseline)        echo "" ;;
    rep0.01)         echo "--loss_scales.rep 0.01" ;;
    free0.1)         echo "--rssm_loss.free 0.1" ;;
    free0)           echo "--rssm_loss.free 0.0" ;;
    free0.1_rep0.01) echo "--rssm_loss.free 0.1 --loss_scales.rep 0.01" ;;
    forces3)         echo "--forces_loss_scale 3.0" ;;
    forces3_wd)      echo "--forces_loss_scale 3.0 --opt.wd 0.1" ;;
    forces3_noise)   echo "--forces_loss_scale 3.0 --forces_aug_noise 0.1" ;;
    all)             echo "--rssm_loss.free 0.1 --loss_scales.rep 0.01 --forces_loss_scale 3.0 --opt.wd 0.1 --forces_aug_noise 0.1" ;;
    *) return 1 ;;
  esac
}

cd "$(dirname "$0")/../.."

for name in $CONFIGS; do
  if ! flags_for "$name" > /dev/null; then
    echo "Unknown setting '$name'. Add it to flags_for() in $0." >&2
    exit 1
  fi
done

if [ "$RATE" -eq 100 ]; then
  TASK=leap_mechanical_pen
else
  TASK=leap_mechanical_pen_${RATE}hz
fi
DATA_DIR=data/leap/${TASK#leap_}
if ! compgen -G "$DATA_DIR/*.npz" > /dev/null; then
  echo "No episodes in $DATA_DIR. Build them first:" >&2
  echo "  python3 data/leap/resample_chunks.py --hz $RATE" >&2
  exit 1
fi

mkdir -p "$SWEEP_DIR"
LOG="$SWEEP_DIR/sweep.log"
log() {
  echo "$(date '+%F %T') $*" | tee -a "$LOG"
}

run_setting() {
  local gpu=$1 name=$2
  local dir="$SWEEP_DIR/$name"
  local flags
  read -r -a flags <<< "$(flags_for "$name")"
  mkdir -p "$dir"
  # The cosine schedule must anneal past the 1000-step warmup.
  local schedule=(--run.steps "$STEPS" --opt.anneal "$((STEPS > 1001 ? STEPS : 1001))")

  if [ ! -f "$dir/model/done" ]; then
    log "$name: training world model on GPU $gpu (${flags[*]:-default flags})"
    if ! LOGDIR="$dir/model" WANDB_NAME="posterior_sweep_${RATE}hz_${name}_model" \
        ./eval/mechanical_pen/train_model.sh "$gpu" "$SEED" "$TASK" "${schedule[@]}" \
        "${flags[@]+"${flags[@]}"}" > "$dir/model.log" 2>&1; then
      log "$name: world model FAILED, see $dir/model.log"
      return
    fi
    touch "$dir/model/done"
  fi
  [ "$MODEL_ONLY" = True ] && { log "$name: done (world model only)"; return; }

  if [ ! -f "$dir/ditto/done" ]; then
    log "$name: training DITTO on GPU $gpu"
    # DITTO gets the default settings, so every setting trains the same policy and only
    # the world model differs. The world-model flags must not be passed: opt.wd would
    # decay the actor and value, which DITTO's optimizer trains.
    if ! LOGDIR="$dir/ditto" MODEL_CHECKPOINT="$dir/model/checkpoint.ckpt" \
        WANDB_NAME="posterior_sweep_${RATE}hz_${name}_ditto" \
        ./eval/mechanical_pen/train_ditto.sh "$gpu" "$SEED" "$TASK" "${schedule[@]}" \
        > "$dir/ditto.log" 2>&1; then
      log "$name: DITTO FAILED, see $dir/ditto.log"
      return
    fi
    touch "$dir/ditto/done"
  fi

  if [ ! -f "$dir/rollout/done" ]; then
    log "$name: rolling out $EPISODES episodes from ${START_SECONDS} s on GPU $gpu"
    if ! EPISODES="$EPISODES" START_SECONDS="$START_SECONDS" \
        ./eval/mechanical_pen/rollout_leq_wm.sh "$gpu" "$dir/ditto" --out "$dir/rollout" \
        > "$dir/rollout.log" 2>&1; then
      log "$name: rollout FAILED, see $dir/rollout.log"
      return
    fi
    touch "$dir/rollout/done"
  fi
  log "$name: done"
}

log "Posterior sweep: settings [$CONFIGS], ${RATE} Hz, GPUs [$GPUS], $STEPS steps," \
  "$EPISODES rollouts (world model only: $MODEL_ONLY)"

read -r -a gpu_list <<< "$GPUS"
read -r -a config_list <<< "$CONFIGS"
pids=()
for g in "${!gpu_list[@]}"; do
  queue=()
  for i in "${!config_list[@]}"; do
    if [ $((i % ${#gpu_list[@]})) -eq "$g" ]; then
      queue+=("${config_list[$i]}")
    fi
  done
  [ ${#queue[@]} -eq 0 ] && continue
  (
    for name in "${queue[@]}"; do
      run_setting "${gpu_list[$g]}" "$name"
    done
  ) &
  pids+=($!)
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done

log "Summarising"
python3 eval/mechanical_pen/posterior_sweep_summary.py --sweep "$SWEEP_DIR" \
  --configs $CONFIGS | tee -a "$LOG"
