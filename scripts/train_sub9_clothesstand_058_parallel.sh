#!/usr/bin/env bash

# Run the three clothes-stand ablations concurrently.
#
# The formal experiments must keep the paper configuration at 4096
# environments.  ``GPU_IDS`` supplies one visible device ID per process.  On
# this host, a measured three-way pressure test with ``GPU_IDS=0,0,0`` peaked
# at 22.2 GiB of the 24 GiB RTX3090, so same-GPU parallelism is supported here;
# separate IDs (for example ``0,1,2``) should be used when additional GPUs are
# available.  Each group has its own IsaacSim process, log, W&B directory, and
# timestamped experiment directory. W&B is online by default;
# override LOGGER_PRESET=logger:wandb_offline only for an explicitly offline run.

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
GPU_IDS="${GPU_IDS:-}"
LAUNCH_DELAY_S="${LAUNCH_DELAY_S:-0}"
LOG_TAG="${LOG_TAG:-parallel}"

if [[ -z "$GPU_IDS" ]]; then
    echo "Error: set GPU_IDS with three device IDs (for example 0,0,0 or 0,1,2)." >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
if [[ "${#gpu_ids[@]}" -lt 3 ]]; then
    echo "Error: GPU_IDS must contain at least three comma-separated GPU IDs (for example 0,1,2)." >&2
    exit 2
fi

ORIGINAL_MOTION="$PROJECT_ROOT/exp/training/$TASK/motions/${TASK}_original_mj_w_obj.npz"
SEMANTIC_B4_MOTION="$PROJECT_ROOT/exp/training/$TASK/motions/${TASK}_semantic_b4_mj_w_obj.npz"
OBJECT_URDF="$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/models/clothesstand/clothesstand.urdf"
SEMANTIC_FILE="$PROJECT_ROOT/exp/training/$TASK/semantic/${TASK}_semantic_adaptive_training.json"
LOG_DIR="$PROJECT_ROOT/exp/training/$TASK/launcher_logs"
MANIFEST="$LOG_DIR/parallel_manifest.txt"

for required_file in "$ORIGINAL_MOTION" "$SEMANTIC_B4_MOTION" "$OBJECT_URDF" "$SEMANTIC_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: required input does not exist: $required_file" >&2
        exit 1
    fi
done

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export CONDA_ENV_NAME
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
source "$PROJECT_ROOT/scripts/source_isaacsim_setup.sh"

{
    echo "task=$TASK"
    echo "started=$(date --iso-8601=seconds)"
    echo "num_envs_per_group=$NUM_ENVS iterations=$TRAINING_ITERATIONS seed=$SEED logger=$LOGGER_PRESET"
    echo "mode=parallel"
} >> "$MANIFEST"

run_one() {
    local group="$1"
    local experiment="$2"
    local run_name="$3"
    local motion_file="$4"
    local gpu_id="$5"
    local log_file="$LOG_DIR/${LOG_TAG}_group${group}.log"
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
    if [[ "$group" == "3" ]]; then
        args+=(
            --command.setup-terms.motion-command.params.motion-config.semantic-file "$SEMANTIC_FILE"
        )
    fi
    if [[ "$LAUNCH_DELAY_S" != "0" ]]; then
        sleep "$(( (group - 1) * LAUNCH_DELAY_S ))"
    fi
    echo "group=$group pid=$BASHPID started=$(date --iso-8601=seconds) log=$log_file tag=$LOG_TAG" >> "$MANIFEST"
    CUDA_VISIBLE_DEVICES="$gpu_id" python src/holosoma/holosoma/train_agent.py "${args[@]}" > "$log_file" 2>&1
    local status="$?"
    echo "group=$group pid=$BASHPID finished=$(date --iso-8601=seconds) exit_status=$status" >> "$MANIFEST"
    return "$status"
}

pids=()
run_one 1 \
    exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
    "${TASK}_originaltraj_originalrl_s${SEED}" \
    "$ORIGINAL_MOTION" "${gpu_ids[0]}" & pids+=("$!")
run_one 2 \
    exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
    "${TASK}_semanticb4traj_originalrl_s${SEED}" \
    "$SEMANTIC_B4_MOTION" "${gpu_ids[1]}" & pids+=("$!")
run_one 3 \
    exp:g1-29dof-wbt-w-object-b4-s2-semantic-adaptive-paper-dr \
    "${TASK}_semanticb4traj_semanticadaptive_s${SEED}" \
    "$SEMANTIC_B4_MOTION" "${gpu_ids[2]}" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one parallel clothes-stand group failed; inspect parallel_group*.log." >&2
    exit "$status"
fi
echo "All three clothes-stand groups completed successfully."
