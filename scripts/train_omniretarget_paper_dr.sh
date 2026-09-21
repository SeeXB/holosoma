#!/usr/bin/env bash

# Launch a B4 paper-DR sampling preset.  All variants use the original Omni
# reward; ``semantic`` is the main semantic-adaptive sampling run.
#
# Usage:
#   ./scripts/train_omniretarget_paper_dr.sh omni|s0 [extra train_agent.py args]
#   ./scripts/train_omniretarget_paper_dr.sh s1 [extra train_agent.py args]
#   ./scripts/train_omniretarget_paper_dr.sh semantic|s2 [extra train_agent.py args]
#
# Environment overrides:
#   NUM_ENVS=2048 TRAINING_ITERATIONS=30000 SEED=42 LOGGER_PRESET=logger:wandb

set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

VARIANT="${1:-}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "$VARIANT" in
    omni|s0)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr"
        RUN_NAME="b4_s0_original_adaptive_paperdr_s${SEED:-42}"
        ;;
    s1)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-b4-s1-semantic-uniform-paper-dr"
        RUN_NAME="b4_s1_semantic_uniform_paperdr_s${SEED:-42}"
        ;;
    semantic|s2)
        EXPERIMENT="exp:g1-29dof-wbt-w-object-b4-s2-semantic-adaptive-paper-dr"
        RUN_NAME="b4_s2_semantic_adaptive_paperdr_s${SEED:-42}"
        ;;
    -h|--help)
        sed -n '3,11p' "$0"
        exit 0
        ;;
    *)
        echo "Error: expected variant 'omni', 's0', 's1', 'semantic', or 's2', got '${VARIANT:-<empty>}'." >&2
        exit 2
        ;;
esac

NUM_ENVS="${NUM_ENVS:-4096}"
TRAINING_ITERATIONS="${TRAINING_ITERATIONS:-30000}"
SEED="${SEED:-42}"
LOGGER_PRESET="${LOGGER_PRESET:-logger:wandb}"
VIDEO_ENABLED="${VIDEO_ENABLED:-False}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"

MOTION_FILE="$PROJECT_ROOT/src/holosoma/holosoma/data/motions/benchmarks/benchmark_results_full_event_transition_truncation/\
transition_truncated_b4_mj_fps50_w_obj.npz"
SEMANTIC_FILE="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/\
semantic_keyframes/sub3_largebox_003_semantic_v2.json"

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

echo "Starting OmniRetarget paper-DR training"
echo "  variant    : $VARIANT"
echo "  experiment : $EXPERIMENT"
echo "  run name   : $RUN_NAME"
echo "  conda env  : $CONDA_ENV_NAME"
echo "  num envs   : $NUM_ENVS"
echo "  iterations : $TRAINING_ITERATIONS"
echo "  seed       : $SEED"
echo "  logger     : $LOGGER_PRESET"
echo "  GPU(s)     : ${CUDA_VISIBLE_DEVICES:-default}"

python src/holosoma/holosoma/train_agent.py \
    "$EXPERIMENT" \
    "$LOGGER_PRESET" \
    --training.name "$RUN_NAME" \
    --training.seed "$SEED" \
    --training.num-envs "$NUM_ENVS" \
    --algo.config.num-learning-iterations "$TRAINING_ITERATIONS" \
    --logger.video.enabled "$VIDEO_ENABLED" \
    --logger.video.upload-to-wandb False \
    "$@"
