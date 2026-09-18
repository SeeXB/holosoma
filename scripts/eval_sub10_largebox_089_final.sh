#!/usr/bin/env bash
# Evaluate the three final task-2 models concurrently with the established
# Paper-DR protocol. Each model is scored on 32 envs x 10 complete episodes.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$PWD"
TASK="sub10_largebox_089"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/exp/eval/$TASK/20260913_final_29999_paper_dr}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-29999}"
NUM_ENVS="${NUM_ENVS:-32}"
SEED="${SEED:-42}"
REFERENCE_DIR="${REFERENCE_DIR:-$PROJECT_ROOT/exp/training/$TASK/motions}"
WANDB_NAME="${WANDB_NAME:-sub10_largebox_089_final_29999_paper_dr_s42}"
EVAL_PARALLEL="${EVAL_PARALLEL:-1}"

RUN1="${RUN1:-$PROJECT_ROOT/logs/WholeBodyTracking/20260907_061352-${TASK}_originaltraj_originalrl_s42_resume8k_v1-locomotion}"
RUN2="${RUN2:-$PROJECT_ROOT/logs/WholeBodyTracking/20260907_061352-${TASK}_semanticb4traj_originalrl_s42_resume8k_v1-locomotion}"
RUN3="${RUN3:-$PROJECT_ROOT/logs/WholeBodyTracking/20260907_061352-${TASK}_semanticb4traj_semanticadaptive_s42_resume8k_v1-locomotion}"

mkdir -p "$EVAL_ROOT"
source scripts/source_isaacsim_setup.sh >/dev/null 2>&1

run_group() {
    local group="$1" run="$2"
    local checkpoint="$run/model_${CHECKPOINT_STEP}.pt"
    local out="$EVAL_ROOT/group${group}_${CHECKPOINT_STEP}"
    local frames fps horizon max_steps
    if [[ ! -s "$checkpoint" ]]; then
        echo "Missing checkpoint: $checkpoint" >&2
        return 1
    fi
    mkdir -p "$out"
    local reference_name
    if [[ "$group" == "1" ]]; then
        reference_name="${TASK}_original_mj_w_obj.npz"
    else
        reference_name="${TASK}_semantic_b4_mj_w_obj.npz"
    fi
    read -r frames fps horizon max_steps < <(python - "$checkpoint" "$REFERENCE_DIR/$reference_name" <<'PY'
import sys
import numpy as np
import torch
from pathlib import Path

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
motion = checkpoint["experiment_config"]["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
if Path(motion).resolve() != Path(sys.argv[2]).resolve():
    raise ValueError(f"checkpoint motion mismatch: {motion} != {sys.argv[2]}")
with np.load(motion, allow_pickle=False) as data:
    frames = len(data["joint_pos"]) - 1
    fps = float(data["fps"].item())
print(frames, fps, frames / fps, 10 * (frames + 2) + 20)
PY
    )
    printf '%s group=%s checkpoint=%s valid_frames=%s fps=%s horizon=%ss max_steps=%s\n' \
        "$(date -Is)" "$group" "$checkpoint" "$frames" "$fps" "$horizon" "$max_steps" \
        > "$out/launch.txt"
    python src/holosoma/holosoma/eval_agent.py \
        --checkpoint "$checkpoint" \
        --import-file scripts/eval_paper_dr_robustness_instrumentation.py \
        --recording.config.enabled \
        --recording.config.output-path "$out/rollout.npz" \
        --eval-overrides.headless True \
        --training.headless True \
        --training.num-envs "$NUM_ENVS" \
        --training.seed "$SEED" \
        --training.max-eval-steps "$max_steps" \
        --training.export-onnx False \
        --simulator.config.sim.max-episode-length-s "$horizon" \
        --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0 \
        --termination.terms.bad-tracking.params.bad-object-pos-threshold 1.0 \
        --termination.terms.bad-tracking.params.bad-object-ori-threshold 0.7853981633974483 \
        --logger.base-dir "$out/logs" \
        --logger.headless-recording False \
        --logger.video.enabled False \
        --logger.video.upload-to-wandb False > "$out/eval.log" 2>&1
    test -s "$out/rollout_all_envs.npz"
    printf '%s group=%s checkpoint=%s complete\n' "$(date -Is)" "$group" "$CHECKPOINT_STEP" \
        >> "$out/launch.txt"
}

if [[ "$EVAL_PARALLEL" == "1" ]]; then
    run_group 1 "$RUN1" & p1=$!
    run_group 2 "$RUN2" & p2=$!
    run_group 3 "$RUN3" & p3=$!
    status=0
    for pid in "$p1" "$p2" "$p3"; do
        wait "$pid" || status=1
    done
    if (( status != 0 )); then
        echo "At least one evaluation failed; inspect $EVAL_ROOT/group*/eval.log." >&2
        exit 1
    fi
else
    run_group 1 "$RUN1"
    run_group 2 "$RUN2"
    run_group 3 "$RUN3"
fi

python scripts/analyze_sub10_largebox_089_final.py \
    --eval-root "$EVAL_ROOT" \
    --checkpoint-step "$CHECKPOINT_STEP" \
    --run-dir "$RUN1" --run-dir "$RUN2" --run-dir "$RUN3" \
    --reference-dir "$REFERENCE_DIR" \
    --wandb-name "$WANDB_NAME" \
    --wandb
