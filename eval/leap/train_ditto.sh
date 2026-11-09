#!/bin/bash -l
# Train DITTO (--algo ditto) on the mechanical pen demos, freezing a trained world model.
#
# Prerequisite: a world model from ./eval/mechanical_pen/train_model.sh [GPU] [SEED] [TASK]
#
# Usage: ./eval/leap/train_ditto.sh [GPU] [SEED] [TASK] [extra main.py flags...]
#   defaults: GPU=0  SEED=0  TASK=leap_mechanical_pen
#
# Loads logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt unless MODEL_CHECKPOINT is set.
# Writes logdir/ditto/<TASK>/<SEED>/ unless LOGDIR is set.
#
# The world model keeps every episode. DITTO's expert is successes only
# (ditto_expert_min_reward). See Ditto_DV3_Implementation.md for why each flag below is set.
#
# W&B (on by default): set WANDB_API_KEY in LEQ_DV3/.env or the environment.
# Override with WANDB_PROJECT / WANDB_NAME, or WANDB_ENABLE=False.
set -euo pipefail

GPU=${1:-0}
SEED=${2:-0}
TASK=${3:-leap_mechanical_pen}
EXTRA_FLAGS=("${@:4}")
SUITE=${SUITE:-mechanical_pen}
ALGO=ditto
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-logdir/model/${TASK}/0/${SEED}/checkpoint.ckpt}
LOGDIR=${LOGDIR:-logdir/$ALGO/$TASK/$SEED}
WANDB_ENABLE=${WANDB_ENABLE:-True}
WANDB_PROJECT=${WANDB_PROJECT:-dreamerv3_LEQ}
WANDB_NAME=${WANDB_NAME:-${ALGO}_${TASK}_${SEED}}
IMAG_LENGTH=${IMAG_LENGTH:-15}
DITTO_GAMMA=${DITTO_GAMMA:-0.95}
ACTENT=${ACTENT:-3e-4}
ACTOR_MINSTD=${ACTOR_MINSTD:-0.01}
ACTOR_MAXSTD=${ACTOR_MAXSTD:-0.1}

cd "$(dirname "$0")/../.."

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
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
  echo "Train it first, e.g. ./eval/mechanical_pen/train_model.sh $GPU $SEED $TASK" >&2
  echo "Or set MODEL_CHECKPOINT to an existing checkpoint.ckpt." >&2
  exit 1
fi

# The world model only works on the force scale it was trained on, and a mismatch
# raises no error of its own: the encoder just sees out-of-range inputs.
python3 - "$SUITE" "$(dirname "$MODEL_CHECKPOINT")/config.yaml" <<'PY'
import pathlib
import sys

from ruamel.yaml import YAML

suite, model_config = sys.argv[1:]
yaml = YAML(typ='safe')
configs = yaml.load(pathlib.Path('dreamerv3/configs.yaml').read_text())
run = dict(configs['defaults']['forces_norm'])
for key, value in configs[suite].items():
  if key.startswith('forces_norm.'):
    run[key.split('.', 1)[1]] = value
path = pathlib.Path(model_config)
if not path.exists():
  sys.exit(f'Cannot check forces_norm: {path} is missing.')
model = yaml.load(path.read_text()).get('forces_norm', {'enable': False})


def same(a, b):
  if bool(a['enable']) != bool(b['enable']):
    return False
  if not a['enable']:
    return True
  return all(
      abs(float(x) - float(y)) < 1e-3
      for key in ('mean', 'std') for x, y in zip(a[key], b[key]))


if not same(run, model):
  sys.exit(
      f'forces_norm mismatch.\n  config "{suite}": {run}\n  checkpoint: {model}\n'
      'The world model would see forces on the wrong scale. Use the config the '
      'checkpoint was trained with.')
print(f'forces_norm matches the checkpoint: {run}')
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  read -r GPU_USED GPU_TOTAL <<< "$(nvidia-smi --query-gpu=memory.used,memory.total \
    --format=csv,noheader,nounits -i "$GPU" | head -1 | tr ',' ' ')"
  if [ -n "${GPU_USED:-}" ] && [ "${GPU_TOTAL:-0}" -gt 0 ]; then
    GPU_PCT=$((GPU_USED * 100 / GPU_TOTAL))
    if [ "$GPU_PCT" -gt 20 ]; then
      echo "Warning: GPU $GPU is ${GPU_PCT}% full (${GPU_USED}/${GPU_TOTAL} MiB)." >&2
      echo "         Another job may be running. Use a free GPU, e.g. ./train_ditto.sh 1 $SEED" >&2
    fi
  fi
fi

set -x
CUDA_VISIBLE_DEVICES=$GPU \
  python3 dreamerv3/main.py \
    --logdir "$LOGDIR" \
    --configs "$SUITE" \
    --task "$TASK" \
    --algo "$ALGO" \
    --seed "$SEED" \
    --run.model_checkpoint "$MODEL_CHECKPOINT" \
    --imag_length "$IMAG_LENGTH" \
    --ditto_gamma "$DITTO_GAMMA" \
    --value.dist mse \
    --actent "$ACTENT" \
    --actor.minstd "$ACTOR_MINSTD" \
    --actor.maxstd "$ACTOR_MAXSTD" \
    --loss_scales.reward 0.0 \
    --loss_scales.replay_critic 0.0 \
    --loss_scales.actor 1.0 \
    --loss_scales.critic 0.0 \
    --loss_scales.value 1.0 \
    --wandb.enable "$WANDB_ENABLE" \
    --wandb.project "$WANDB_PROJECT" \
    --wandb.name "$WANDB_NAME" \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"
