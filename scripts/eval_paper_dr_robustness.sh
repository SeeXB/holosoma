#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT LABEL OUTPUT_NPZ" >&2
  exit 2
fi

checkpoint=$1
label=$2
output_npz=$3
output_npz=$(realpath -m "$output_npz")

source scripts/source_isaacsim_setup.sh >/dev/null 2>&1

python src/holosoma/holosoma/eval_agent.py \
  --checkpoint "$checkpoint" \
  --import-file scripts/eval_paper_dr_robustness_instrumentation.py \
  --recording.config.enabled \
  --recording.config.output-path "$output_npz" \
  --eval-overrides.headless True \
  --training.headless True \
  --training.num-envs 32 \
  --training.seed 42 \
  --training.max-eval-steps 3300 \
  --training.export-onnx False \
  --simulator.config.sim.max-episode-length-s 6.48 \
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale 0.0 \
  --termination.terms.bad-tracking.params.bad-object-pos-threshold 1.0 \
  --termination.terms.bad-tracking.params.bad-object-ori-threshold 0.7853981633974483 \
  --logger.headless-recording False \
  --logger.video.enabled False \
  --logger.video.upload-to-wandb False
