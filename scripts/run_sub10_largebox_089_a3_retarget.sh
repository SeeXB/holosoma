#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RETARGET_ROOT="${PROJECT_ROOT}/src/holosoma_retargeting"
PACKAGE_DIR="${RETARGET_ROOT}/holosoma_retargeting"
PYTHON_BIN="${PYTHON_BIN:-/home/zongyouyu/miniconda3/envs/hsretargeting/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/exp/retargeting/a3_sub10_largebox_089_20260921}"
TASK="sub10_largebox_089"
INPUT_ROOT="${PACKAGE_DIR}/demo_data/omomo/official_inputs"
SCENE="${PACKAGE_DIR}/demo_data/models/a3/a3_31dof_w_largebox.xml"
PLAN="${PACKAGE_DIR}/demo_data/semantic_keyframes/frozen/dynamic_json_object_rotate_v5_semantic_b4/omomo/${TASK}/semantic_plan.json"

export PYTHONPATH="${RETARGET_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "${OUTPUT_ROOT}/original" "${OUTPUT_ROOT}/semantic_b4"
cd "${PACKAGE_DIR}"

common=(
  --task-type object_interaction
  --robot a3
  --task-name "${TASK}"
  --data-format smplh
  --data-path "${INPUT_ROOT}"
  --task-config.object-name largebox
  --task-config.scene-xml-file "${SCENE}"
  --retargeter.foot-sticking-tolerance 0.02
)

"${PYTHON_BIN}" -m holosoma_retargeting.examples.robot_retarget \
  "${common[@]}" \
  --save-dir "${OUTPUT_ROOT}/original" \
  --semantic.mode original \
  --semantic.profile-dir "${OUTPUT_ROOT}/original"

"${PYTHON_BIN}" -m holosoma_retargeting.examples.robot_retarget \
  "${common[@]}" \
  --save-dir "${OUTPUT_ROOT}/semantic_b4" \
  --semantic.mode uniform2_semantic_weight_full_event_transition_truncated_budget \
  --semantic.exact-trigger-budget 4 \
  --semantic.semantic-keyframe-path "${PLAN}" \
  --semantic.profile-dir "${OUTPUT_ROOT}/semantic_b4"

mv -f \
  "${OUTPUT_ROOT}/semantic_b4/${TASK}_original.npz" \
  "${OUTPUT_ROOT}/semantic_b4/${TASK}_semantic_b4.npz"

echo "Original:    ${OUTPUT_ROOT}/original/${TASK}_original.npz"
echo "Semantic B4: ${OUTPUT_ROOT}/semantic_b4/${TASK}_semantic_b4.npz"
