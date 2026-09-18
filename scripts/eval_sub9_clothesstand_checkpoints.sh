#!/usr/bin/env bash
# Three groups in parallel, with checkpoints evaluated serially within a group.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PROJECT_ROOT="$PWD"
EVAL_ROOT="${EVAL_ROOT:-$PROJECT_ROOT/exp/eval/sub9_clothesstand_058/20260905_headless_isolated_8k_12k}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-08000 12000}"
mkdir -p "$EVAL_ROOT"
source scripts/source_isaacsim_setup.sh >/dev/null 2>&1

run_group() {
    local group="$1" suffix="$2" step checkpoint out frames horizon max_steps
    local run="$PROJECT_ROOT/logs/WholeBodyTracking/20260903_093625-sub9_clothesstand_058_${suffix}_s42-locomotion"
    for step in $CHECKPOINT_STEPS; do
        checkpoint="$run/model_${step}.pt"
        out="$EVAL_ROOT/group${group}_${step}"
        mkdir -p "$out"
        # Derive the clip duration from the checkpoint's own reference, not the
        # previously evaluated largebox clip. No reference override is applied.
        read -r frames horizon max_steps < <(python - "$checkpoint" <<'PY'
import sys
import numpy as np
import torch
c = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = c["experiment_config"]["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
with np.load(m, allow_pickle=False) as z:
    frames = len(z["joint_pos"]) - 1
    fps = float(z["fps"].item())
print(frames, frames / fps, 10 * (frames + 2) + 20)
PY
        )
        printf '%s group=%s checkpoint=%s valid_frames=%s horizon=%ss\n' "$(date -Is)" "$group" "$checkpoint" "$frames" "$horizon"
        python src/holosoma/holosoma/eval_agent.py \
            --checkpoint "$checkpoint" \
            --import-file scripts/eval_paper_dr_robustness_instrumentation.py \
            --recording.config.enabled \
            --recording.config.output-path "$out/rollout.npz" \
            --eval-overrides.headless True \
            --training.headless True \
            --training.num-envs 32 \
            --training.seed 42 \
            --training.max-eval-steps "$max_steps" \
            --training.export-onnx False \
            --simulator.config.sim.max-episode-length-s "$horizon" \
            --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0 \
            --termination.terms.bad-tracking.params.bad-object-pos-threshold 1.0 \
            --termination.terms.bad-tracking.params.bad-object-ori-threshold 0.7853981633974483 \
            --logger.base-dir "$out/logs" \
            --logger.headless-recording False \
            --logger.video.enabled False \
            --logger.video.upload-to-wandb False >"$out/eval.log" 2>&1
        test -s "$out/rollout_all_envs.npz"
        printf '%s group=%s checkpoint=%s complete\n' "$(date -Is)" "$group" "$step"
    done
}

run_group 1 originaltraj_originalrl & p1=$!
run_group 2 semanticb4traj_originalrl & p2=$!
run_group 3 semanticb4traj_semanticadaptive & p3=$!
status=0
for pid in "$p1" "$p2" "$p3"; do
    wait "$pid" || status=1
done
if (( status != 0 )); then
    echo 'At least one evaluation failed; inspect the per-checkpoint eval.log.' >&2
    exit 1
fi
python scripts/analyze_clothesstand_checkpoints.py --eval-root "$EVAL_ROOT" --wandb
