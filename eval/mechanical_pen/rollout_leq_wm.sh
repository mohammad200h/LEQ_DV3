#!/bin/bash -l
# Roll a trained mechanical pen policy out inside its world model and render GIFs.
# Same as eval/leap/rollout_leq_wm.sh, with defaults for the mechanical pen: 20 Hz data.
# No sim or hand needed.
#
# Usage: ./eval/mechanical_pen/rollout_leq_wm.sh [GPU] [LOGDIR] [extra rollout_leq_wm.py flags...]
#   defaults: GPU=0  LOGDIR=logdir/ditto/leap_mechanical_pen_20hz/0
#
# For each held-out success, the world model watches the real episode up to --start
# (default: START_SECONDS=0.8 s at the data rate), then:
#   Posterior        keeps filtering the real episode: real obs + the expert's previous action
#                    each step. The policy is queried on every posterior state, never executed.
#   Prior, expert    imagines the rest with the expert's actions (the world model's own error)
#   Prior, policy    imagines the rest with the policy's actions (as at deployment)
#
# rollout.gif shows Real | Posterior | Prior expert | Prior policy hands, coloured by
# ||force||, above two jitter traces: command change per step (expert, policy on posterior,
# policy in prior) and decoded taxel movement per step. If "policy on posterior" is as
# jumpy as "policy in prior", the jitter is the policy's own; if only the prior is jumpy,
# it comes from imagination.
#
# Output in <LOGDIR>/rollout_wm/<episode>/ (or --out): rollout.gif, summary.png,
# rollout.npz, metrics.json.
#
# Settings (environment variables):
#   EPISODES=3          Held-out successes to roll out (--num-episodes overrides it)
#   START_SECONDS=0.8   Real time observed before imagining (--start overrides it)
#   FPS=10              GIF frame rate; the GIF plays in real time
#
# Examples:
#   ./eval/mechanical_pen/rollout_leq_wm.sh 1
#   ./eval/mechanical_pen/rollout_leq_wm.sh 1 logdir/rate_sweep/50hz/ditto
#   ./eval/mechanical_pen/rollout_leq_wm.sh 1 logdir/ditto/leap_mechanical_pen_20hz/0 --outcome fail
set -euo pipefail

GPU=${1:-0}
LOGDIR=${2:-logdir/ditto/leap_mechanical_pen_20hz/0}
EXTRA_FLAGS=("${@:3}")
EPISODES=${EPISODES:-3}
START_SECONDS=${START_SECONDS:-0.8}
FPS=${FPS:-10}

cd "$(dirname "$0")/../.."

if [ ! -f "$LOGDIR/checkpoint.ckpt" ]; then
  echo "No checkpoint in $LOGDIR. Train one first, e.g. ./eval/mechanical_pen/train_ditto.sh" >&2
  exit 1
fi

# The data rate comes from the run's task, e.g. leap_mechanical_pen_50hz -> 50.
TASK=$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["task"])' \
  "$LOGDIR/config.yaml")
DATA="data/leap/${TASK#leap_}"
DATA="${DATA%%-*}"
HZ=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["freq"])' \
  "$DATA/manifest.json" 2>/dev/null || echo 100)
START=$(python3 -c "print(max(1, round($START_SECONDS * $HZ)))")

# Flags given on the command line come last, so they win.
./eval/leap/rollout_leq_wm.sh "$GPU" "$LOGDIR" \
  --num-episodes "$EPISODES" --start "$START" --fps "$FPS" \
  "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
