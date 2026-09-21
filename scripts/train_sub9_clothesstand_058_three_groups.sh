#!/usr/bin/env bash

# Train the three clothes-stand ablations requested for the first OMOMO task.
# The groups are deliberately sequential because this host has one 24-GB GPU.
#
# Usage:
#   NUM_ENVS=4096 TRAINING_ITERATIONS=30000 SEED=42 \
#     ./scripts/train_sub9_clothesstand_058_three_groups.sh
#
# Useful overrides:
#   LOGGER_PRESET=logger:disabled   # local smoke/debug run
#   START_GROUP=2                   # resume from group 2 or 3
#   STOP_GROUP=2                    # run only one group
#   QUIET=True                      # write iteration output only to group*.log
#   CONDA_ENV_NAME=hssim

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TASK="sub9_clothesstand_058"

NUM_ENVS="${NUM_ENVS:-4096}"
TRAINING_ITERATIONS="${TRAINING_ITERATIONS:-30000}"
SEED="${SEED:-42}"
LOGGER_PRESET="${LOGGER_PRESET:-logger:wandb}"
VIDEO_ENABLED="${VIDEO_ENABLED:-False}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"
START_GROUP="${START_GROUP:-1}"
STOP_GROUP="${STOP_GROUP:-3}"

ORIGINAL_MOTION="$PROJECT_ROOT/src/holosoma/holosoma/data/motions/tasks/$TASK/${TASK}_original_mj_w_obj.npz"
SEMANTIC_B4_MOTION="$PROJECT_ROOT/src/holosoma/holosoma/data/motions/tasks/$TASK/${TASK}_semantic_b4_mj_w_obj.npz"
OBJECT_URDF="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/clothesstand/clothesstand.urdf"
SEMANTIC_FILE="$PROJECT_ROOT/src/holosoma/holosoma/data/semantic/$TASK/${TASK}_semantic_adaptive_training.json"
LOG_DIR="$PROJECT_ROOT/exp/training/$TASK/launcher_logs"
MANIFEST="$LOG_DIR/three_groups_manifest.txt"

for required_file in "$ORIGINAL_MOTION" "$SEMANTIC_B4_MOTION" "$OBJECT_URDF" "$SEMANTIC_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: required input does not exist: $required_file" >&2
        exit 1
    fi
done
if ! [[ "$START_GROUP" =~ ^[1-3]$ && "$STOP_GROUP" =~ ^[1-3]$ && "$START_GROUP" -le "$STOP_GROUP" ]]; then
    echo "Error: START_GROUP/STOP_GROUP must satisfy 1 <= START_GROUP <= STOP_GROUP <= 3" >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export CONDA_ENV_NAME
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

{
    echo "task=$TASK"
    echo "started=$(date --iso-8601=seconds)"
    echo "num_envs=$NUM_ENVS iterations=$TRAINING_ITERATIONS seed=$SEED logger=$LOGGER_PRESET"
    echo "original_motion=$ORIGINAL_MOTION"
    echo "semantic_b4_motion=$SEMANTIC_B4_MOTION"
    echo "object_urdf=$OBJECT_URDF"
    echo "semantic_file=$SEMANTIC_FILE"
} >> "$MANIFEST"

run_group() {
    local group="$1"
    local experiment="$2"
    local run_name="$3"
    local motion_file="$4"
    local use_semantic_file="${5:-False}"
    local log_file="$LOG_DIR/group${group}.log"

    echo
    echo "============================================================"
    echo "Starting group $group: $run_name"
    echo "  experiment: $experiment"
    echo "  motion:     $motion_file"
    echo "  semantic:   $use_semantic_file"
    echo "  log:        $log_file"
    echo "============================================================"

    local args=(
        "$experiment"
        "$LOGGER_PRESET"
        --training.headless True
        --training.name "$run_name"
        --training.seed "$SEED"
        --training.num-envs "$NUM_ENVS"
        --algo.config.num-learning-iterations "$TRAINING_ITERATIONS"
        --training.export-onnx True
        --logger.video.enabled "$VIDEO_ENABLED"
        --logger.video.upload-to-wandb False
        --scene.rigid-objects.object.urdf-file "$OBJECT_URDF"
        --command.setup-terms.motion-command.params.motion-config.motion-file "$motion_file"
    )
    if [[ "$use_semantic_file" == "True" ]]; then
        args+=(
            --command.setup-terms.motion-command.params.motion-config.semantic-file "$SEMANTIC_FILE"
        )
    fi

    local status=0
    set +e
    if [[ "${QUIET:-False}" == "True" ]]; then
        python src/holosoma/holosoma/train_agent.py "${args[@]}" > "$log_file" 2>&1
        status="$?"
    else
        python src/holosoma/holosoma/train_agent.py "${args[@]}" 2>&1 | tee "$log_file"
        status="${PIPESTATUS[0]}"
    fi
    set -e
    echo "group=$group run=$run_name finished=$(date --iso-8601=seconds) exit_status=$status" >> "$MANIFEST"
    if [[ "$status" -ne 0 ]]; then
        echo "Group $group failed with exit status $status; later groups were not started." >&2
        exit "$status"
    fi
}

if [[ "$START_GROUP" -le 1 && "$STOP_GROUP" -ge 1 ]]; then
    run_group 1 \
        exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
        "${TASK}_originaltraj_originalrl_s${SEED}" \
        "$ORIGINAL_MOTION"
fi
if [[ "$START_GROUP" -le 2 && "$STOP_GROUP" -ge 2 ]]; then
    run_group 2 \
        exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
        "${TASK}_semanticb4traj_originalrl_s${SEED}" \
        "$SEMANTIC_B4_MOTION"
fi
if [[ "$START_GROUP" -le 3 && "$STOP_GROUP" -ge 3 ]]; then
    run_group 3 \
        exp:g1-29dof-wbt-w-object-b4-s2-semantic-adaptive-paper-dr \
        "${TASK}_semanticb4traj_semanticadaptive_s${SEED}" \
        "$SEMANTIC_B4_MOTION" \
        True
fi

echo "All requested clothes-stand groups completed successfully."
