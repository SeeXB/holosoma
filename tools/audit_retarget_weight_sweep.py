#!/usr/bin/env python3
"""Compare contact, existing precision and raw temporal smoothness together."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from audit_retarget_contact_comparison import audit as contact_audit
from holosoma_retargeting.semantic_keyframes.contact_targets import load_retarget_contact_targets


def statistics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {key: None for key in ("rms", "mean_abs", "p95_abs", "max_abs")}
    return {
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "mean_abs": float(np.mean(np.abs(values))),
        "p95_abs": float(np.percentile(np.abs(values), 95)),
        "max_abs": float(np.max(np.abs(values))),
    }


def active_spans(active: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(active, dtype=bool), False]
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(start), int(stop - 1)) for start, stop in boundaries.reshape(-1, 2)]


def joint_smoothness(
    angles_rad: np.ndarray, fps: float, spans: list[tuple[int, int]],
) -> dict:
    """Do not differentiate across gaps between separate contact spans."""
    samples = {key: [] for key in ("step_deg", "speed", "acceleration", "jerk")}
    reversal_count = 0
    jump_frames = 0
    for start, end in spans:
        step = np.diff(np.rad2deg(angles_rad[start:end + 1]), axis=0)
        speed = step * fps
        acceleration = np.diff(speed, axis=0) * fps
        jerk = np.diff(acceleration, axis=0) * fps
        for key, values in zip(samples, (step, speed, acceleration, jerk)):
            if values.size:
                samples[key].append(values)
        reversal_count += int(np.sum(
            (step[:-1] * step[1:] < 0)
            & (np.abs(step[:-1]) >= .2) & (np.abs(step[1:]) >= .2)
        ))
        jump_frames += int(np.any(np.abs(step) > 5, axis=1).sum())
    return {
        **{key: statistics(np.concatenate(values) if values else np.empty(0))
           for key, values in samples.items()},
        "back_and_forth_steps_gt_0p2deg": reversal_count,
        "frame_count_any_joint_step_gt_5deg": jump_frames,
        "spans_inclusive": spans,
    }


def evaluate(args: argparse.Namespace) -> dict:
    labels = {}
    for spec in args.trajectory:
        label, separator, path = spec.partition("=")
        if not separator or not label or label in labels:
            raise ValueError("--trajectory requires unique LABEL=PATH entries")
        labels[label] = Path(path).resolve()
    contacts = load_retarget_contact_targets(args.contacts)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    hinges = np.flatnonzero(model.jnt_type == int(mujoco.mjtJoint.mjJNT_HINGE))
    indices = model.jnt_qposadr[hinges]
    names = [model.joint(int(joint)).name for joint in hinges]
    spans = active_spans(contacts.active.all(axis=1))
    reference_path = next(iter(labels.values()))
    reference_qpos = reference_fps = None
    rows = {}
    for label, path in labels.items():
        with np.load(path, allow_pickle=False) as data:
            qpos = np.asarray(data["qpos"], dtype=float)
            fps = float(data["fps"])
        if (qpos.shape != (len(contacts.active), model.nq)
                or not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0
                or not 0 <= args.late_start < len(qpos) - 3):
            raise ValueError(f"invalid trajectory dimensions/FPS/late segment: {path}")
        if reference_qpos is None:
            reference_qpos, reference_fps = qpos, fps
        elif fps != reference_fps or not np.array_equal(reference_qpos[:, -7:], qpos[:, -7:]):
            raise ValueError("object trajectory or FPS differs between runs")
        pair = contact_audit(SimpleNamespace(
            original=reference_path, final=path, model=args.model,
            contacts=args.contacts, plan=args.plan,
        ))
        row = pair["rows"]["optimized"]
        angles = qpos[:, indices]
        segments = {"all": [(0, len(qpos) - 1)], "contact": spans,
                    "late": [(args.late_start, len(qpos) - 1)]}
        local_window = getattr(args, "local_window", None)
        if local_window is not None:
            start, end = local_window
            if not 0 <= start < end < len(qpos) or end - start < 3:
                raise ValueError("local window must contain at least four frames inside trajectory")
            segments["local"] = [(int(start), int(end))]
        row["joint_smoothness"] = {
            name: joint_smoothness(angles, fps, selected)
            for name, selected in segments.items()
        }
        arm_indices = [index for index, name in enumerate(names)
                       if any(token in name for token in ("shoulder_", "elbow_", "wrist_"))]
        if arm_indices:
            row["arm_joint_smoothness"] = {
                name: joint_smoothness(angles[:, arm_indices], fps, selected)
                for name, selected in segments.items()
            }
        step = np.diff(np.rad2deg(angles), axis=0)
        maximum = np.max(np.abs(step), axis=1)
        row["largest_steps"] = [
            {"from_frame": int(frame), "to_frame": int(frame + 1),
             "joint": names[int(np.argmax(np.abs(step[frame])))],
             "signed_step_deg": float(step[frame, np.argmax(np.abs(step[frame]))])}
            for frame in np.argsort(maximum)[::-1][:10]
        ]
        row["contact_onset_steps"] = [
            {"from_frame": start - 1, "to_frame": start,
             "max_joint_step_deg": float(maximum[start - 1]),
             "joint": names[int(np.argmax(np.abs(step[start - 1])))]}
            for start, _ in spans if start > 0
        ]
        metrics_path = path.with_name("metrics.json")
        recorded = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        row["physical_metrics"] = {
            key: recorded.get(key) for key in (
                "max_body_box_penetration_mm", "penetration_fraction",
                "sliding_fraction", "contact_preservation",
            )
        }
        job_path = path.with_name("job.json")
        row["job"] = json.loads(job_path.read_text()) if job_path.exists() else None
        rows[label] = row
    return {
        "definitions": {
            "precision_mm": "same plan, source geometry and adjacency; unweighted old metrics",
            "smoothness": "raw hinge angles, no filtering; RMS over frames and joints",
            "speed_unit": "deg/s", "acceleration_unit": "deg/s^2", "jerk_unit": "deg/s^3",
            "contact_spans_inclusive": spans, "late_start_frame": args.late_start,
            "hand_gap": "minimum signed distance of selected hand collision mesh; not palm-center gap",
            "palm_anchor_error": "fixed palm center to target anchor, full 3D distance",
            "playback": "kinematic trajectory, not an RL or physical-grasp success-rate test",
        },
        "checks": {"frames": len(reference_qpos), "fps": reference_fps,
                   "hinge_joints": len(hinges), "object_qpos_equal": True,
                   "source_geometry_and_adjacency_equal": True},
        "inputs": {"model": str(args.model.resolve()), "plan": str(args.plan.resolve()),
                   "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(),
                   "contacts": str(args.contacts.resolve()),
                   "contacts_sha256": hashlib.sha256(args.contacts.read_bytes()).hexdigest()},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", action="append", required=True, metavar="LABEL=NPZ")
    for name in ("model", "contacts", "plan", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--late-start", type=int, default=90)
    parser.add_argument("--local-window", type=int, nargs=2, metavar=("START", "END"))
    args = parser.parse_args()
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rows": list(result["rows"]), "checks": result["checks"],
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
