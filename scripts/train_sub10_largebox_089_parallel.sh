#!/usr/bin/env bash

# Run the three task-2 ablations concurrently with the registered Paper-DR
# experiment presets. W&B online logging and 4096 environments are defaults.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TASK="sub10_largebox_089"

NUM_ENVS="${NUM_ENVS:-4096}"
TRAINING_ITERATIONS="${TRAINING_ITERATIONS:-30000}"
SEED="${SEED:-42}"
LOGGER_PRESET="${LOGGER_PRESET:-logger:wandb}"
VIDEO_ENABLED="${VIDEO_ENABLED:-False}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"
GPU_IDS="${GPU_IDS:-}"
LAUNCH_DELAY_S="${LAUNCH_DELAY_S:-0}"
LOG_TAG="${LOG_TAG:-parallel}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
MONITOR_INTERVAL_S="${MONITOR_INTERVAL_S:-15}"
GROUP1_CHECKPOINT="${GROUP1_CHECKPOINT:-}"
GROUP2_CHECKPOINT="${GROUP2_CHECKPOINT:-}"
GROUP3_CHECKPOINT="${GROUP3_CHECKPOINT:-}"

if [[ -z "$GPU_IDS" ]]; then
    echo "Error: set GPU_IDS with three device IDs (for this host: GPU_IDS=0,0,0)." >&2
    exit 2
fi
IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
if [[ "${#gpu_ids[@]}" -lt 3 ]]; then
    echo "Error: GPU_IDS must contain at least three comma-separated IDs." >&2
    exit 2
fi

ORIGINAL_MOTION="${ORIGINAL_MOTION:-$PROJECT_ROOT/src/holosoma/holosoma/data/motions/tasks/$TASK/${TASK}_original_mj_w_obj.npz}"
SEMANTIC_B4_MOTION="${SEMANTIC_B4_MOTION:-$PROJECT_ROOT/src/holosoma/holosoma/data/motions/tasks/$TASK/${TASK}_semantic_b4_mj_w_obj.npz}"
OBJECT_URDF="${OBJECT_URDF:-$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/largebox/largebox.urdf}"
SEMANTIC_FILE="${SEMANTIC_FILE:-$PROJECT_ROOT/src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles/$TASK/semantic_keyframes/${TASK}_dynamic.json}"
LOG_DIR="$PROJECT_ROOT/exp/training/$TASK/launcher_logs"
MANIFEST="$LOG_DIR/parallel_manifest.txt"
RESOURCE_LOG="$LOG_DIR/${LOG_TAG}_resources.csv"

for required_file in "$ORIGINAL_MOTION" "$SEMANTIC_B4_MOTION" "$OBJECT_URDF" "$SEMANTIC_FILE"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Error: required input does not exist: $required_file" >&2
        exit 1
    fi
done
for checkpoint in "$GROUP1_CHECKPOINT" "$GROUP2_CHECKPOINT" "$GROUP3_CHECKPOINT"; do
    if [[ -n "$checkpoint" && ! -f "$checkpoint" ]]; then
        echo "Error: checkpoint does not exist: $checkpoint" >&2
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
    echo "original_motion=$ORIGINAL_MOTION"
    echo "semantic_b4_motion=$SEMANTIC_B4_MOTION"
    echo "semantic_file=$SEMANTIC_FILE"
    echo "group1_checkpoint=${GROUP1_CHECKPOINT:-none}"
    echo "group2_checkpoint=${GROUP2_CHECKPOINT:-none}"
    echo "group3_checkpoint=${GROUP3_CHECKPOINT:-none}"
} >> "$MANIFEST"

monitor_resources() {
    echo "timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct,system_memory_used_mib,system_memory_total_mib" > "$RESOURCE_LOG"
    while true; do
        local timestamp system_used system_total
        timestamp="$(date --iso-8601=seconds)"
        read -r system_total system_used < <(free -m | awk 'NR == 2 {print $2, $3}')
        gpu_rows="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits 2>/dev/null || true)"
        while IFS= read -r gpu_row; do
            [[ -z "$gpu_row" ]] && continue
            echo "$timestamp,$gpu_row,$system_used,$system_total" >> "$RESOURCE_LOG"
        done <<< "$gpu_rows"
        sleep "$MONITOR_INTERVAL_S"
    done
}

run_one() {
    local group="$1"
    local experiment="$2"
    local run_name="$3"
    local motion_file="$4"
    local gpu_id="$5"
    local checkpoint="$6"
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
    if [[ -n "$checkpoint" ]]; then
        args+=(--training.checkpoint "$checkpoint")
    fi
    if [[ "$LAUNCH_DELAY_S" != "0" ]]; then
        sleep "$(( (group - 1) * LAUNCH_DELAY_S ))"
    fi
    echo "group=$group pid=$BASHPID started=$(date --iso-8601=seconds) log=$log_file tag=$LOG_TAG" >> "$MANIFEST"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu_id" python src/holosoma/holosoma/train_agent.py "${args[@]}" > "$log_file" 2>&1
    local status="$?"
    set -e
    echo "group=$group pid=$BASHPID finished=$(date --iso-8601=seconds) exit_status=$status" >> "$MANIFEST"
    return "$status"
}

monitor_resources &
monitor_pid="$!"
cleanup_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

pids=()
run_one 1 \
    exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
    "${TASK}_originaltraj_originalrl_s${SEED}${RUN_SUFFIX}" \
    "$ORIGINAL_MOTION" "${gpu_ids[0]}" "$GROUP1_CHECKPOINT" & pids+=("$!")
run_one 2 \
    exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
    "${TASK}_semanticb4traj_originalrl_s${SEED}${RUN_SUFFIX}" \
    "$SEMANTIC_B4_MOTION" "${gpu_ids[1]}" "$GROUP2_CHECKPOINT" & pids+=("$!")
run_one 3 \
    exp:g1-29dof-wbt-w-object-b4-s2-semantic-adaptive-paper-dr \
    "${TASK}_semanticb4traj_semanticadaptive_s${SEED}${RUN_SUFFIX}" \
    "$SEMANTIC_B4_MOTION" "${gpu_ids[2]}" "$GROUP3_CHECKPOINT" & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "At least one task-2 group failed; inspect ${LOG_TAG}_group*.log." >&2
    exit "$status"
fi
echo "All three task-2 groups completed successfully."
