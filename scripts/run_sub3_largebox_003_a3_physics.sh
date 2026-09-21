#!/usr/bin/env bash
# Reproduce the selected MuJoCo physics validation for both A3 retargets.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/source_retargeting_setup.sh >/dev/null 2>&1

RETARGET_DIR="exp/retargeting/a3_sub3_largebox_003_20260922"
OUTPUT_DIR="${OUTPUT_DIR:-exp/physics/a3_sub3_largebox_003_mujoco/final}"

MUJOCO_GL="${MUJOCO_GL:-egl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/mujoco_a3_box_lift_search.py \
  --mode replay \
  --motion both \
  --original-motion "${RETARGET_DIR}/original/sub3_largebox_003_original.npz" \
  --semantic-b4-motion "${RETARGET_DIR}/semantic_b4/sub3_largebox_003_semantic_b4.npz" \
  --dimensions 0.30 0.34 0.34 \
  --ground-height -0.054 \
  --settle-seconds 0.15 \
  --tail-seconds 0.75 \
  --box-friction 1.2 \
  --contact-latch \
  --contact-latch-mode object_reference \
  --grasp-reference-blend-seconds 0.75 \
  --grip-start-frame 20 \
  --latch-release-frame 170 \
  --align-box-to-lowest-hand-line \
  --render \
  --output "${OUTPUT_DIR}" \
  "$@"

ffmpeg -loglevel error -y \
  -i "${OUTPUT_DIR}/original/physics_replay.mp4" \
  -i "${OUTPUT_DIR}/semantic_b4/physics_replay.mp4" \
  -filter_complex \
  "[0:v]drawtext=text='Original OmniRetarget':x=24:y=24:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[left];[1:v]drawtext=text='Semantic B4':x=24:y=24:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  "${OUTPUT_DIR}/original_vs_semantic_b4_physics.mp4"
