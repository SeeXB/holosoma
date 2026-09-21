#!/usr/bin/env bash
# Reproduce the final MuJoCo physics validation for both A3 retargets.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/source_retargeting_setup.sh >/dev/null 2>&1

OUTPUT_DIR="${OUTPUT_DIR:-exp/physics/a3_sub10_largebox_089_mujoco/final_g1_pose_place}"

MUJOCO_GL="${MUJOCO_GL:-egl}" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/mujoco_a3_box_lift_search.py \
  --mode replay \
  --motion both \
  --dimensions 0.47115421 0.475 0.40789548 \
  --ground-height -0.066 \
  --settle-seconds 0.15 \
  --tail-seconds 0.75 \
  --box-friction 1.2 \
  --contact-latch \
  --contact-latch-mode object_reference \
  --grasp-reference-blend-seconds 0.75 \
  --grip-start-frame 4 \
  --latch-release-frame 125 \
  --align-box-to-lowest-hand-line \
  --render \
  --output "$OUTPUT_DIR" \
  "$@"

ffmpeg -loglevel error -y \
  -i "$OUTPUT_DIR/original/physics_replay.mp4" \
  -i "$OUTPUT_DIR/semantic_b4/physics_replay.mp4" \
  -filter_complex \
  "[0:v]drawtext=text='Original OmniRetarget':x=24:y=24:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[left];[1:v]drawtext=text='Semantic B4':x=24:y=24:fontsize=28:fontcolor=white:borderw=2:bordercolor=black[right];[left][right]hstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p \
  "$OUTPUT_DIR/original_vs_semantic_b4_physics.mp4"
