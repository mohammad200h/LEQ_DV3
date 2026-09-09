#!/bin/bash -l
# Sweep LEQ over the grid in a search config, one run at a time.
#
# Prerequisite: ./eval/leap/train_model.sh [GPU] [SEED] [TASK] for every task in
# the grid, so that logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt exists.
#
# Usage: ./eval/leap/run_search.sh [CONFIG]
#   default: CONFIG=eval/leap/search.yaml
#
# The config takes `tasks` and `expectiles` lists plus scalar `gpu` and `seed`;
# the sweep is their cross product. Each point calls ./eval/leap/train_leq.sh,
# so it writes logdir/leq/<TASK>/<EXPECTILE>/<SEED>/ and gets its own W&B run
# named leq_<TASK>_<EXPECTILE>_<SEED>.
#
# DRY_RUN=True    print the commands without training.
# WANDB_PROJECT / WANDB_ENABLE are passed through to every run.
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG=${1:-eval/leap/search.yaml}
DRY_RUN=${DRY_RUN:-False}

if [ ! -f "$CONFIG" ]; then
  echo "Missing search config: $CONFIG" >&2
  exit 1
fi

# train_leq.sh derives WANDB_NAME per run, but only if it is unset -- an
# inherited value would put every point of the sweep in one W&B run.
unset WANDB_NAME

GRID=$(python3 - "$CONFIG" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path) as f:
  cfg = yaml.safe_load(f) or {}
if not isinstance(cfg, dict):
  sys.exit(f'{path}: expected a mapping at the top level')
tasks = cfg.get('tasks') or []
expectiles = cfg.get('expectiles') or []
if not tasks:
  sys.exit(f'{path}: no "tasks" list')
if not expectiles:
  sys.exit(f'{path}: no "expectiles" list')
print(cfg.get('gpu', 0))
print(cfg.get('seed', 0))
for task in tasks:
  for expectile in expectiles:
    print(f'{task} {expectile}')
PY
)

mapfile -t GRID_LINES <<< "$GRID"
GPU=${GRID_LINES[0]}
SEED=${GRID_LINES[1]}
RUNS=("${GRID_LINES[@]:2}")

# Fail before the first run rather than partway through the sweep.
for RUN in "${RUNS[@]}"; do
  read -r TASK _ <<< "$RUN"
  CKPT="logdir/model/${TASK}/0/${SEED}/checkpoint.ckpt"
  if [ ! -f "$CKPT" ]; then
    echo "Missing world-model checkpoint for $TASK: $CKPT" >&2
    echo "Train it first: ./eval/leap/train_model.sh $GPU $SEED $TASK" >&2
    exit 1
  fi
done

echo "Search config: $CONFIG"
echo "GPU $GPU, seed $SEED, ${#RUNS[@]} run(s):"
for RUN in "${RUNS[@]}"; do
  read -r TASK EXPECTILE <<< "$RUN"
  echo "  ./eval/leap/train_leq.sh $GPU $SEED $TASK $EXPECTILE"
done

if [ "$DRY_RUN" = "True" ]; then
  echo "DRY_RUN=True, nothing launched."
  exit 0
fi

FAILED=()
INDEX=0
for RUN in "${RUNS[@]}"; do
  read -r TASK EXPECTILE <<< "$RUN"
  INDEX=$((INDEX + 1))
  echo
  echo "=== [$INDEX/${#RUNS[@]}] $TASK expectile=$EXPECTILE seed=$SEED ==="
  START=$SECONDS
  if ./eval/leap/train_leq.sh "$GPU" "$SEED" "$TASK" "$EXPECTILE"; then
    echo "=== [$INDEX/${#RUNS[@]}] done in $((SECONDS - START))s ==="
  else
    STATUS=$?
    echo "=== [$INDEX/${#RUNS[@]}] FAILED (exit $STATUS) after $((SECONDS - START))s ===" >&2
    FAILED+=("$TASK expectile=$EXPECTILE")
  fi
done

echo
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "${#FAILED[@]} of ${#RUNS[@]} run(s) failed:" >&2
  for RUN in "${FAILED[@]}"; do
    echo "  $RUN" >&2
  done
  exit 1
fi
echo "All ${#RUNS[@]} run(s) finished."
