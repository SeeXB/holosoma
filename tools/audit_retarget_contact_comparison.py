#!/usr/bin/env python3
"""Compare geometry and existing precision metrics under one frozen plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from render_retarget_contact_comparison import (
    PART_BODIES, _body_id, _collision_geoms, _load_qpos, _object_geoms,
    _set_qpos, _surface_distance,
    _palm_measurements,
)
from holosoma_retargeting.semantic_keyframes.contact_targets import load_retarget_contact_targets
from holosoma_retargeting.semantic_keyframes.precision import evaluate_precision, load_precision_payload
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection


def audit(args: argparse.Namespace) -> dict:
    contacts = load_retarget_contact_targets(args.contacts)
    plan = load_semantic_plan_projection(args.plan)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    object_geoms = _object_geoms(model)
    robot_geoms = {
        part: _collision_geoms(model, _body_id(model, PART_BODIES[part]))
        for part in contacts.parts
    }
    part_bodies = {part: _body_id(model, PART_BODIES[part]) for part in contacts.parts}
    object_body = int(model.geom_bodyid[object_geoms[0]])
    rows = {}
    reference = None
    for label, path in (("original", args.original), ("optimized", args.final)):
        qpos, fps = _load_qpos(path)
        contacts.validate(len(qpos))
        if qpos.shape[1] != model.nq:
            raise ValueError("trajectory/model dimension mismatch")
        payload = load_precision_payload(path)
        if reference is None:
            reference = (payload.source_vertices, payload.adjacency, qpos.shape, fps)
        elif not (
            np.array_equal(reference[0], payload.source_vertices)
            and np.array_equal(reference[1], payload.adjacency)
            and reference[2:] == (qpos.shape, fps)
        ):
            raise ValueError("comparison source geometry/adjacency/frames/FPS differ")
        precision = evaluate_precision(payload, plan, critical_event_names=None)
        data = mujoco.MjData(model)
        signed = np.empty(contacts.active.shape)
        palm_rows = []
        for frame, q in enumerate(qpos):
            _set_qpos(model, data, q)
            for index, part in enumerate(contacts.parts):
                signed[frame, index] = _surface_distance(model, data, robot_geoms[part], object_geoms)
            palm_rows.append(_palm_measurements(data, object_body, part_bodies, contacts, frame))
        parts = {}
        for index, part in enumerate(contacts.parts):
            values = signed[contacts.active[:, index], index]
            gap = np.maximum(values, 0.0)
            parts[part] = {
                "active_frames": len(values),
                "mean_gap_mm": float(gap.mean() * 1000),
                "median_gap_mm": float(np.median(gap) * 1000),
                "p95_gap_mm": float(np.percentile(gap, 95) * 1000),
                "max_gap_mm": float(gap.max() * 1000),
                "within_2cm_fraction": float(np.mean(gap <= 0.02)),
                "max_overlap_mm": float(max(0.0, -values.min()) * 1000),
                "overlap_gt_1mm_frames": int(np.sum(values < -0.001)),
                "overlap_gt_10mm_frames": int(np.sum(values < -0.01)),
            }
            if palm_rows[0]:
                selected = [palm_rows[frame][part] for frame in np.flatnonzero(contacts.active[:, index])]
                angles = np.array([row["normal_error_deg"] for row in selected])
                anchors = np.array([row["anchor_error_m"] for row in selected])
                parts[part]["palm_normal_mean_deg"] = float(angles.mean())
                parts[part]["palm_normal_p95_deg"] = float(np.percentile(angles, 95))
                parts[part]["palm_normal_max_deg"] = float(angles.max())
                parts[part]["palm_anchor_mean_error_mm"] = float(anchors.mean() * 1000)
                parts[part]["palm_anchor_max_error_mm"] = float(anchors.max() * 1000)
                parts[part]["palm_at_frame121"] = palm_rows[121][part] if len(palm_rows) > 121 else None
        joint_frames = contacts.active.all(axis=1)
        rows[label] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "precision_mm": {
                "global_exact": float(1000 * precision.keyframe_global["exact"]),
                "semantic_part_exact": float(1000 * precision.semantic_part["exact"]),
                "local_exact": float(1000 * precision.semantic_local["exact"]),
                "body_object_edge_exact": float(1000 * precision.semantic_edge["exact"]),
                "ordinary": float(1000 * precision.ordinary["mean"]),
                "all_frame_unweighted_mean": float(1000 * payload.residuals.mean()),
            },
            "contacts": parts,
            "all_declared_parts_within_2cm_fraction": float(
                np.mean((signed[joint_frames] <= 0.02).all(axis=1))
            ),
            "all_declared_parts_within_2cm_and_no_gt_1mm_overlap_fraction": float(
                np.mean(((signed[joint_frames] <= 0.02) & (signed[joint_frames] >= -0.001)).all(axis=1))
            ),
        }
    return {
        "comparison": "same source geometry, adjacency, keyframe windows and contact evaluation frames",
        "playback": "kinematic qpos replay, not an RL rollout or physical grasp validation",
        "plan": str(args.plan.resolve()),
        "contact_activation_evidence": contacts.metadata["activation_evidence"],
        "contact_spans": {
            part: [[span["start_frame"], span["end_frame"]] for span in contacts.metadata["active_spans"][part]]
            for part in contacts.parts
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "final", "model", "contacts", "plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
