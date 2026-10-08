#!/usr/bin/env bash
# Activate the IsaacSim/HSSim environment before running this script.
# Resume the original-reward OmniRetarget +10 cm run from iteration 1500.
# PPO counts num-learning-iterations as additional iterations: 1500 + 28500.
set -euo pipefail

task_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_repo_dir"

task_python="${A3_PYTHON:-python}"
task_checkpoint="${A3_CHECKPOINT:-$task_repo_dir/logs/WholeBodyTracking/20261001_165721-a3_sub3_largebox_003_omni_original_fwd10_originalrl_s42_20261002-locomotion/model_01500.pt}"
task_motion="${A3_MOTION_FILE:-$task_repo_dir/exp/retargeting/a3_sub3_largebox_003_omni_original_20261001/sub3_largebox_003_a3_omni_original_flat_fwd10cm.npz}"
task_asset="$task_repo_dir/src/holosoma_retargeting/holosoma_retargeting/demo_data/models/a3/a3_31dof.urdf"

for task_file in "$task_checkpoint" "$task_motion" "$task_asset"; do
  if [[ ! -f "$task_file" ]]; then
    printf 'Missing required file: %s\nCopy the A3 assets, reference and checkpoint with rsync first.\n' "$task_file" >&2
    exit 1
  fi
done

export PYTHONPATH="$task_repo_dir/src/holosoma${PYTHONPATH:+:$PYTHONPATH}"
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES="${A3_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export WANDB_MODE=offline

exec "$task_python" src/holosoma/holosoma/train_agent.py \
  exp:a3-31dof-wbt-w-object-b4-s0-original-adaptive-paper-dr \
  logger:wandb_offline \
  --training.headless True \
  --training.name a3_omni_original_fwd10_originalrl_resume1500_s42 \
  --training.seed 42 \
  --training.num-envs "${A3_NUM_ENVS:-512}" \
  --training.checkpoint "$task_checkpoint" \
  --algo.config.num-learning-iterations "${A3_ADDITIONAL_ITERATIONS:-28500}" \
  --algo.config.save-interval 500 \
  --training.export-onnx True \
  --logger.video.enabled False \
  --logger.video.upload-to-wandb False \
  --command.setup-terms.motion-command.params.motion-config.motion-file "$task_motion"
