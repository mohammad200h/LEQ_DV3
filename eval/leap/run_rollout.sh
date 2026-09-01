#!/bin/bash -l
# Roll out a trained LEQ policy in sparsh-skin-sim LeapFlexEnv.
#
# Usage: ./eval/leap/run_rollout.sh [GPU] [LOGDIR] [ENV_CONFIG] [EPISODES] [--render]
#   defaults:
#     GPU=0
#     LOGDIR=logdir/leq/leap_squeeze/0.1/1
#     ENV_CONFIG=/workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml
#     EPISODES=1
set -euo pipefail

GPU=${1:-0}
LOGDIR=${2:-logdir/leq/leap_squeeze/0.1/1}
ENV_CONFIG=${3:-/workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml}
EPISODES=${4:-1}
RENDER_FLAG=()
if [ "${5:-}" = "--render" ] || [ "${5:-}" = "render" ]; then
  RENDER_FLAG=(--render)
fi

cd "$(dirname "$0")/../.."

if [ -f /workspace/LEQ/docker/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /workspace/LEQ/docker/.env
  set +a
fi

XLA_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.75}
export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"

echo "CUDA_VISIBLE_DEVICES=$GPU python3 eval/leap/rollout_leq.py --logdir $LOGDIR --env-config $ENV_CONFIG --episodes $EPISODES ${RENDER_FLAG[*]:-}"
CUDA_VISIBLE_DEVICES="$GPU" \
  python3 eval/leap/rollout_leq.py \
    --logdir "$LOGDIR" \
    --env-config "$ENV_CONFIG" \
    --episodes "$EPISODES" \
    "${RENDER_FLAG[@]}"
