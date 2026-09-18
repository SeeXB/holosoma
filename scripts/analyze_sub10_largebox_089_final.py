"""Analyze the three final sub10_largebox_089 Paper-DR evaluations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_paper_dr_robustness import (  # noqa: E402
    DISPLAY_PRIORITY,
    _clustered_ci,
    _trajectory_errors,
    episode_rows,
)


ROOT = Path(__file__).resolve().parents[1]
TASK = "sub10_largebox_089"
GROUPS = {
    1: (
        "Original trajectory + original RL",
        "20260907_061352-sub10_largebox_089_originaltraj_originalrl_s42_resume8k_v1-locomotion",
        "sub10_largebox_089_original_mj_w_obj.npz",
    ),
    2: (
        "Semantic B4 trajectory + original RL",
        "20260907_061352-sub10_largebox_089_semanticb4traj_originalrl_s42_resume8k_v1-locomotion",
        "sub10_largebox_089_semantic_b4_mj_w_obj.npz",
    ),
    3: (
        "Semantic B4 trajectory + semantic adaptive RL",
        "20260907_061352-sub10_largebox_089_semanticb4traj_semanticadaptive_s42_resume8k_v1-locomotion",
        "sub10_largebox_089_semantic_b4_mj_w_obj.npz",
    ),
}


def summarize_run(
    directory: Path,
    quota: int,
    checkpoint_step: int,
    run_dirs: dict[int, Path] | None = None,
    reference_dir: Path | None = None,
    reference_file: Path | None = None,
) -> dict:
    group = int(directory.name.split("_", 1)[0].removeprefix("group"))
    label, run_name, reference_name = GROUPS[group]
    reference = reference_file or (reference_dir or ROOT / "exp" / "training" / TASK / "motions") / reference_name
    run_dir = run_dirs[group] if run_dirs else ROOT / "logs" / "WholeBodyTracking" / run_name
    checkpoint = run_dir / f"model_{checkpoint_step:05d}.pt"
    if not checkpoint.is_file() or not reference.is_file():
        raise FileNotFoundError(f"missing checkpoint or reference: {checkpoint}, {reference}")
    recording = directory / "rollout_all_envs.npz"
    with np.load(reference, allow_pickle=False) as data:
        valid_frames = len(data["joint_pos"]) - 1
        fps = float(data["fps"].item())

    rows, metadata = episode_rows(recording, quota=quota)
    for row in rows:
        row["reached"] = valid_frames if row["timeout"] else row["motion_step"]
    num_envs = int(metadata["num_envs"])
    failures = [row for row in rows if not row["success"]]
    failure_reasons = Counter(
        next((key for key in DISPLAY_PRIORITY if key in row["flags"]), row["classification"])
        for row in failures
    )
    successes = sum(bool(row["success"]) for row in rows)

    with np.load(recording, allow_pickle=False) as data:
        starts = [int(data["motion_step"][row["start_step"], row["env"]]) for row in rows]
        durations = [row["terminal_step"] - row["start_step"] + 1 for row in rows]
        if max(starts) > 1:
            raise ValueError(f"{recording}: evaluation did not start at frame 0: {sorted(set(starts))}")
        pushes = 0
        if "push_count" in data.files:
            for env in range(num_envs):
                last = max(row["terminal_step"] for row in rows if row["env"] == env)
                pushes += int(data["push_count"][last, env])

    ci_low, ci_high = _clustered_ci(rows, num_envs)
    return {
        "group": group,
        "label": label,
        "checkpoint_iteration": checkpoint_step,
        "checkpoint": str(checkpoint),
        "reference": str(reference),
        "recording": str(recording),
        "num_envs": num_envs,
        "episodes_per_env": quota,
        "episodes_scored": len(rows),
        "successes": successes,
        "failures": len(rows) - successes,
        "success_rate": successes / len(rows),
        "clustered_95ci": [max(0.0, ci_low), min(1.0, ci_high)],
        "valid_motion_frames": valid_frames,
        "motion_fps": fps,
        "horizon_seconds": valid_frames / fps,
        "start_motion_steps_observed": sorted(set(starts)),
        "mean_episode_seconds": float(np.mean(durations) / fps),
        "mean_reached_frame": float(np.mean([row["reached"] for row in rows])),
        "median_reached_frame": float(np.median([row["reached"] for row in rows])),
        "max_reached_frame": max(row["reached"] for row in rows),
        "exclusive_failure_reasons": dict(failure_reasons),
        "raw_failure_flags": dict(Counter(key for row in failures for key in row["flags"])),
        "stale_timeout_terminal_flags": dict(
            Counter(key for row in rows for key in row["terminal_flags"] if key not in row["flags"])
        ),
        "success_by_episode": [
            sum(bool(row["success"]) for row in rows if row["episode"] == episode)
            for episode in range(1, quota + 1)
        ],
        "success_by_env": [
            sum(bool(row["success"]) for row in rows if row["env"] == env)
            for env in range(num_envs)
        ],
        "trajectory_errors": _trajectory_errors(recording, rows),
        "push_events_in_scored_episodes": pushes,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=29999)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--run-dir", type=Path, action="append", help="run directory for each group, in 1/2/3 order")
    parser.add_argument("--reference-dir", type=Path, help="directory containing the two reference motions")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-name", default="sub10_largebox_089_final_29999_paper_dr_s42")
    args = parser.parse_args()

    if args.run_dir is not None and len(args.run_dir) != 3:
        parser.error("--run-dir must be given exactly three times (groups 1, 2, 3)")
    run_dirs = dict(enumerate(args.run_dir, start=1)) if args.run_dir else None
    paths = sorted(path.parent for path in args.eval_root.glob("group*_*/rollout_all_envs.npz"))
    if len(paths) != 3:
        parser.error(f"expected 3 completed recordings, found {len(paths)}")
    results = [
        summarize_run(path, args.quota, args.checkpoint_step, run_dirs, args.reference_dir)
        for path in paths
    ]
    frames = {result["valid_motion_frames"] for result in results}
    fps_values = {result["motion_fps"] for result in results}
    if len(frames) != 1 or len(fps_values) != 1:
        raise ValueError(f"evaluation references have inconsistent lengths/fps: {frames}, {fps_values}")
    valid_frames = frames.pop()
    fps = fps_values.pop()
    protocol = (
        "Paper-DR, seed 42, 32 environments, first 10 complete episodes each "
        f"(320/model), full {valid_frames}-frame horizon ({valid_frames / fps:.2f} s), initial-pose noise off, "
        "Paper-DR pushes on, object thresholds 1 m/45 deg. Group 1 uses Original; "
        "groups 2/3 use B4. Shape scaling is unsupported."
    )
    report = {"task": TASK, "protocol": protocol, "results": results}
    (args.eval_root / "final_results.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# sub10_largebox_089 final checkpoint evaluation",
        "",
        protocol,
        "",
        f"| Group | SR | Successes | 95% CI | Mean duration (s) | Mean reached / {valid_frames} | Object RMSE (m) | Orientation RMSE (deg) | Body RMSE (m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        errors = result["trajectory_errors"]
        ci = result["clustered_95ci"]
        lines.append(
            f"| {result['group']}: {result['label']} | {100 * result['success_rate']:.2f}% | "
            f"{result['successes']}/{result['episodes_scored']} | "
            f"{100 * ci[0]:.2f}%–{100 * ci[1]:.2f}% | {result['mean_episode_seconds']:.3f} | "
            f"{result['mean_reached_frame']:.1f} | {errors['object_pos_rmse_m']:.4f} | "
            f"{errors['object_ori_rmse_deg']:.2f} | {errors['tracked_body_pos_rmse_m']:.4f} |"
        )
    for result in results:
        lines.extend(
            [
                "",
                f"## Group {result['group']}",
                "",
                f"Checkpoint: `{result['checkpoint']}`",
                "",
                f"Failure reasons: `{result['exclusive_failure_reasons']}`",
                "",
                f"Success by episode: `{result['success_by_episode']}`",
                "",
                f"Push events: {result['push_events_in_scored_episodes']}",
            ]
        )
    (args.eval_root / "FINAL_EVAL_REPORT.md").write_text("\n".join(lines) + "\n")

    for result in results:
        print(
            f"group={result['group']} SR={result['successes']}/{result['episodes_scored']} "
            f"duration={result['mean_episode_seconds']:.3f}s "
            f"reasons={result['exclusive_failure_reasons']}"
        )

    if args.wandb:
        import wandb

        with wandb.init(
            entity="yumou0319-",
            project="WholeBodyTracking",
            mode="online",
            name=args.wandb_name,
            group="sub10_largebox_089_final_eval",
            dir=str(args.eval_root),
            config={
                "task": TASK,
                "protocol": protocol,
                "seed": 42,
                "num_envs": 32,
                "episodes_per_env": args.quota,
                "checkpoint_iteration": args.checkpoint_step,
            },
        ) as run:
            metrics = {"checkpoint_iteration": args.checkpoint_step}
            for result in results:
                prefix = f"group{result['group']}"
                values = {
                    key: result[key]
                    for key in (
                        "success_rate",
                        "successes",
                        "episodes_scored",
                        "mean_episode_seconds",
                        "mean_reached_frame",
                        "push_events_in_scored_episodes",
                    )
                }
                values.update(result["trajectory_errors"])
                metrics.update({f"{prefix}/{key}": value for key, value in values.items()})
            run.log(metrics)
            (args.eval_root / "wandb_url.txt").write_text(run.url + "\n")
            print(f"W&B: {run.url}")


if __name__ == "__main__":
    main()
