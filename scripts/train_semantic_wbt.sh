#!/usr/bin/env bash

# Launch the controlled Semantic Keyframe Preference WBT experiments.
#
# Usage:
#   ./scripts/train_semantic_wbt.sh [r0|r1|r2|r3|r4] [additional train_agent.py arguments]
#
# Examples:
#   ./scripts/train_semantic_wbt.sh
#   NUM_ENVS=1024 ./scripts/train_semantic_wbt.sh r4
#   TRAINING_ITERATIONS=1000 ./scripts/train_semantic_wbt.sh r2 logger:wandb

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VARIANT="${1:-r4}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$VARIANT" in
    r0|u2)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-semantic-r0-u2-omni"
        ;;
    r1|b4|e0|base)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-semantic-r1-b4-omni"
        ;;
    r2|e1|part)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-semantic-r2-b4-part"
        ;;
    r3|e2|part-rel)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-semantic-r3-b4-part-rel"
        ;;
    r4|e3|semantic|full)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-semantic-r4-b4-full"
        ;;
    -h|--help)
        sed -n '3,11p' "$0"
        exit 0
        ;;
    *)
        echo "Error: unknown variant '$VARIANT'; expected r0, r1, r2, r3, or r4." >&2
        exit 2
        ;;
esac

NUM_ENVS="${NUM_ENVS:-4096}"
TRAINING_ITERATIONS="${TRAINING_ITERATIONS:-30000}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"

MOTION_FILE="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/rl/transition_truncated_b4_mj_fps50_w_obj.npz"
SEMANTIC_FILE="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json"

for required_file in "$MOTION_FILE" "$SEMANTIC_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: required input does not exist: $required_file" >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
export CONDA_ENV_NAME
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

echo "Starting Semantic WBT training"
echo "  experiment : $EXPERIMENT"
echo "  conda env  : $CONDA_ENV_NAME"
echo "  num envs   : $NUM_ENVS"
echo "  iterations : $TRAINING_ITERATIONS"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-default}"

python src/holosoma/holosoma/train_agent.py \
    "$EXPERIMENT" \
    --training.num-envs "$NUM_ENVS" \
    --algo.config.num-learning-iterations "$TRAINING_ITERATIONS" \
    "$@"
