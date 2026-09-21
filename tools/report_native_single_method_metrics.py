#!/usr/bin/env python3
"""Report native penetration metrics for a single-method retargeting batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_std(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    return float(values.mean()), float(values.std(ddof=0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", default="semantic_b4")
    args = parser.parse_args()
    run = args.run.resolve()
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = [job for job in manifest["jobs"] if job["method"] == args.method]
    if not jobs:
        raise ValueError(f"No {args.method!r} jobs in {manifest_path}")

    rows = []
    failed = []
    all_depths_cm: list[float] = []
    for job in jobs:
        task_run = Path(job["run"])
        metrics_path = task_run / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("status") != "ok":
            failed.append({"task": job["task"], "error": metrics.get("error")})
            continue
        raw_path = task_run / "native_penetration_raw.json"
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        duration = float(raw["duration"])
        if not np.isclose(duration, float(metrics["penetration_fraction"]), atol=1e-12, rtol=0):
            raise ValueError(f"Duration mismatch: {raw_path}")
        depths_cm = [100.0 * float(value) for value in raw["penetration_max_depths_m"]]
        all_depths_cm.extend(depths_cm)
        trajectory = Path(metrics["result"])
        rows.append(
            {
                "task": job["task"],
                "frames": int(metrics["frames"]),
                "trajectory": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "activate_obj_non_penetration": bool(job.get("activate_obj_non_penetration", True)),
                "penetration_duration": duration,
                "violating_frames": len(depths_cm),
                "violating_frame_depth_cm_mean": float(np.mean(depths_cm)) if depths_cm else 0.0,
                "violating_frame_depth_cm_max": max(depths_cm, default=0.0),
            }
        )

    durations = np.asarray([row["penetration_duration"] for row in rows], dtype=np.float64)
    depths = np.asarray(all_depths_cm, dtype=np.float64)
    duration_mean, duration_std = mean_std(durations)
    depth_mean, depth_std = mean_std(depths)
    payload = {
        "schema": "holosoma.native_single_method_metrics.v1",
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "method": args.method,
        "successful_trajectories": len(rows),
        "failed_trajectories": failed,
        "penetration_tolerance_m": 0.01,
        "aggregation": {
            "duration": "mean/std over per-trajectory frame fractions",
            "max_depth": "concatenate per-violating-frame maximum depths, then mean/std; no zero padding",
            "std_ddof": 0,
        },
        "summary": {
            "duration_mean": duration_mean,
            "duration_std": duration_std,
            "max_depth_cm_mean": depth_mean,
            "max_depth_cm_std": depth_std,
            "absolute_peak_depth_cm": float(depths.max()) if depths.size else 0.0,
            "violating_frames": int(depths.size),
            "total_frames": int(sum(row["frames"] for row in rows)),
        },
        "rows": rows,
    }
    json_path = run / f"native_{args.method}_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        f"# Native penetration metrics: {args.method}",
        "",
        f"成功轨迹：{len(rows)}/{len(jobs)}；阈值：10 mm。",
        "",
        "| Duration mean ± std | Max Depth mean ± std (cm) | Absolute peak (cm) | Violating frames / total |",
        "|---:|---:|---:|---:|",
        f"| {duration_mean:.6f} ± {duration_std:.6f} | {depth_mean:.4f} ± {depth_std:.4f} | "
        f"{payload['summary']['absolute_peak_depth_cm']:.4f} | {len(depths)}/{payload['summary']['total_frames']} |",
        "",
        "Duration 对每条轨迹的穿透帧比例取 mean/std。Max Depth 将所有超过 10 mm 的帧最大深度拼接后取 mean/std，不为无穿透任务补零。",
        "",
        "| Task | Duration | Violating frames | Mean depth (cm) | Peak depth (cm) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['penetration_duration']:.6f} | {row['violating_frames']} | "
            f"{row['violating_frame_depth_cm_mean']:.4f} | {row['violating_frame_depth_cm_max']:.4f} |"
        )
    for row in failed:
        lines.append(f"| {row['task']} | failed | — | — | — |")
    report_path = run / f"NATIVE_{args.method.upper()}_METRICS.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(report_path)


if __name__ == "__main__":
    main()
