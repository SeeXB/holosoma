"""Read-only audit of OMOMO -> InterMimic -> retarget object-pose ordering.

Writes diagnostics only. Does not regenerate or alter any experiment inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from prepare_batch_retarget_inputs import TASK_OBJECTS, canonicalize_object_poses
from holosoma_retargeting.src.utils import load_intermimic_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT / "exp/retargeting/omomo_batch/input")
    parser.add_argument("--runs-root", type=Path, default=ROOT / "exp/retargeting/omomo_batch/runs")
    args = parser.parse_args()
    manifest_path = args.input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = {row["task_name"]: row for row in manifest.get("tasks", [])}
    rows = []
    for task in TASK_OBJECTS:
        with np.load(ROOT / f"exp/omomo_cari4d/{task}/input/omomo_gt_sequence.npz", allow_pickle=False) as z:
            gt = z["object_poses_wxyz_xyz"].copy()
            gt_human = z["human_joints"].copy()
            raw_gt = gt.copy()
            alignment = None
            if manifest.get("schema") == "holosoma.omomo_batch_inputs.v3":
                gt, alignment = canonicalize_object_poses(
                    gt, z["object_vertices_local"], z["object_scale"],
                    trimesh.load_mesh(entries[task]["mesh"], process=False).vertices)
        path = args.input_root / f"{task}.pt"
        full = torch.load(path, map_location="cpu", weights_only=False).numpy()
        packed = full[:, 318:325]
        reader_error = None
        try:
            human, loaded = load_intermimic_data(str(path))
        except ValueError as error:
            # Forensic inspection of rejected legacy data ONLY. Never feeds a
            # bypassed validation result to retargeting or training.
            reader_error = str(error)
            human = full[:, 162:318].reshape(-1, 52, 3)
            loaded = packed[:, [6, 3, 4, 5, 0, 1, 2]]
        # Actual reader consumes xyz + xyzw. This inverse is demonstrated in
        # memory only; production files are deliberately left untouched.
        corrected_packed = gt[:, [4, 5, 6, 1, 2, 3, 0]]
        corrected_roundtrip = corrected_packed[:, [6, 3, 4, 5, 0, 1, 2]]
        np.testing.assert_array_equal(corrected_roundtrip, gt)
        np.testing.assert_allclose(loaded, packed[:, [6, 3, 4, 5, 0, 1, 2]], atol=0)
        dist = np.linalg.norm(loaded[:, 4:] - gt[:, 4:], axis=-1)
        qnorm = np.linalg.norm(loaded[:, :4], axis=-1)
        artifacts = []
        for method in ("original", "uniform2", "semantic_b4"):
            p = args.runs_root / method / task / f"{task}_{method}.npz"
            if not p.exists():
                continue
            with np.load(p, allow_pickle=False) as z:
                q = z["qpos"]
                artifacts.append({"method": method, "path": str(p), "frames": len(q),
                                  "max_quat_difference_from_malformed_input":
                                      float(np.max(np.abs(q[:, -4:] - loaded[:len(q), :4]))),
                                  "raw_object_quaternion_norm_range":
                                      [float(np.linalg.norm(q[:, -4:], axis=-1).min()),
                                       float(np.linalg.norm(q[:, -4:], axis=-1).max())]})
        rows.append({"task": task, "frames": len(gt),
                     "canonical_alignment": alignment, "first_raw_gt_wxyz_xyz": raw_gt[0].tolist(),
                     "reader_validation_error": reader_error,
                     "human_joints_max_abs_error": float(np.max(np.abs(human - gt_human))),
                     "object_pose_roundtrip_max_abs_error": float(np.max(np.abs(loaded - gt))),
                     "translation_error_before_retarget_scaling_m": {
                         "first": float(dist[0]), "min": float(dist.min()), "max": float(dist.max())},
                     "loaded_quaternion_norm_range": [float(qnorm.min()), float(qnorm.max())],
                     "first_gt_wxyz_xyz": gt[0].tolist(), "first_loaded_wxyz_xyz": loaded[0].tolist(),
                     "corrected_in_memory_roundtrip_max_abs_error": 0.0,
                     "retarget_artifacts": artifacts})
    report = {"schema": "omomo_pose_roundtrip_diagnostic_v3", "read_only": True,
              "input_root": str(args.input_root), "runs_root": str(args.runs_root),
              "affected_tasks": sum(r["object_pose_roundtrip_max_abs_error"] > 1e-6 for r in rows),
              "tasks_checked": len(rows), "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"affected_tasks": report["affected_tasks"], "tasks_checked": len(rows),
                      "retarget_artifacts_with_same_bad_quaternion": sum(
                          a["max_quat_difference_from_malformed_input"] < 1e-6
                          for r in rows for a in r["retarget_artifacts"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
