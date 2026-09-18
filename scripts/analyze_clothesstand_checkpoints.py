"""Score task-specific checkpoints with the established first-10 Paper-DR rule.

This task has 208 valid frames, not the largebox evaluator's 324. Keep the
existing terminal-snapshot correction and RMSE calculation, but do not import
the largebox-specific phase names or survival horizon.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyze_paper_dr_robustness import (
    DISPLAY_PRIORITY, _clustered_ci, _trajectory_errors, episode_rows,
)

GROUPS = {
    1: ("Original trajectory + original RL", "originaltraj_originalrl"),
    2: ("Semantic B4 trajectory + original RL", "semanticb4traj_originalrl"),
    3: ("Semantic B4 trajectory + semantic adaptive RL", "semanticb4traj_semanticadaptive"),
}
ROOT = Path(__file__).resolve().parents[1]


def summarize_run(directory: Path, quota: int) -> dict:
    group_text, step_text = directory.name.split("_")
    group, step = int(group_text.removeprefix("group")), int(step_text)
    label, suffix = GROUPS[group]
    reference = ROOT / "exp/training/sub9_clothesstand_058/motions" / (
        "sub9_clothesstand_058_original_mj_w_obj.npz" if group == 1
        else "sub9_clothesstand_058_semantic_b4_mj_w_obj.npz"
    )
    with np.load(reference, allow_pickle=False) as z:
        frames = len(z["joint_pos"]) - 1
        fps = float(z["fps"].item())
    recording = directory / "rollout_all_envs.npz"
    rows, metadata = episode_rows(recording, quota=quota)
    for row in rows:
        row["reached"] = frames if row["timeout"] else row["motion_step"]
    nenvs = int(metadata["num_envs"])
    failures = [row for row in rows if not row["success"]]
    reasons = Counter(next((key for key in DISPLAY_PRIORITY if key in row["flags"]), row["classification"])
                      for row in failures)
    successes = sum(row["success"] for row in rows)
    with np.load(recording, allow_pickle=False) as z:
        starts = [int(z["motion_step"][row["start_step"], row["env"]]) for row in rows]
        durations = [row["terminal_step"] - row["start_step"] + 1 for row in rows]
        if max(starts) > 1:
            raise ValueError(f"{recording}: evaluation did not consistently start at frame 0: {sorted(set(starts))}")
        pushes = 0
        for env in range(nenvs):
            last = max(row["terminal_step"] for row in rows if row["env"] == env)
            pushes += int(z["push_count"][last, env])
    checkpoint = ROOT / "logs/WholeBodyTracking" / (
        f"20260903_093625-sub9_clothesstand_058_{suffix}_s42-locomotion"
    ) / f"model_{step:05d}.pt"
    return {
        "group": group, "label": label, "checkpoint_iteration": step,
        "checkpoint": str(checkpoint), "reference": str(reference),
        "recording": str(recording), "num_envs": nenvs,
        "episodes_per_env": quota, "episodes_scored": len(rows),
        "successes": successes, "success_rate": successes / len(rows),
        "clustered_95ci": list(_clustered_ci(rows, nenvs)),
        "valid_motion_frames": frames, "motion_fps": fps,
        "horizon_seconds": frames / fps,
        "start_motion_steps_observed": sorted(set(starts)),
        "mean_episode_seconds": float(np.mean(durations) / fps),
        "mean_reached_frame": float(np.mean([row["reached"] for row in rows])),
        "median_reached_frame": float(np.median([row["reached"] for row in rows])),
        "max_reached_frame": max(row["reached"] for row in rows),
        "exclusive_failure_reasons": dict(reasons),
        "raw_failure_flags": dict(Counter(key for row in failures for key in row["flags"])),
        "stale_timeout_terminal_flags": dict(Counter(key for row in rows for key in row["terminal_flags"] if key not in row["flags"])),
        "success_by_env": [sum(row["success"] for row in rows if row["env"] == env) for env in range(nenvs)],
        "trajectory_errors": _trajectory_errors(recording, rows),
        "push_events_in_scored_episodes": pushes,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--wandb", action="store_true", help="Upload scalar evaluation metrics, without raw recordings or checkpoints")
    args = parser.parse_args()
    paths = sorted(p.parent for p in args.eval_root.glob("group*_*/rollout_all_envs.npz"))
    if not paths:
        parser.error("No completed recordings found")
    results = [summarize_run(path, args.quota) for path in paths]
    report = {
        "task": "sub9_clothesstand_058",
        "note": "No 10k checkpoint was saved. 8k and 12k bracket it; neither is an exact 10k result.",
        "protocol": "Paper-DR, seed 42, 32 environments, first 10 complete episodes each, full 208-frame horizon (4.16s), initial-pose noise off, paper pushes on, object thresholds 1m/45deg. Group 1 uses its Original reference; groups 2/3 use their B4 reference. Shape scaling unsupported.",
        "results": results,
    }
    (args.eval_root / "checkpoint_results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Clothesstand checkpoint evaluation", "", report["note"], "", report["protocol"], "",
             "RMSE covers the observed valid frames of the scored episodes, including early failures; it does not imply full-clip completion.", "",
             "| Group | Iteration | SR | Successes | Mean duration (s) | Mean reached frame / 208 | Object RMSE (m) | Orientation RMSE (deg) | Body RMSE (m) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        e = r["trajectory_errors"]
        lines.append(f"| {r['group']}: {r['label']} | {r['checkpoint_iteration']} | {100*r['success_rate']:.2f}% | {r['successes']}/{r['episodes_scored']} | {r['mean_episode_seconds']:.3f} | {r['mean_reached_frame']:.1f} | {e['object_pos_rmse_m']:.4f} | {e['object_ori_rmse_deg']:.2f} | {e['tracked_body_pos_rmse_m']:.4f} |")
        print(f"group={r['group']} iter={r['checkpoint_iteration']} SR={r['successes']}/{r['episodes_scored']} duration={r['mean_episode_seconds']:.3f}s reasons={r['exclusive_failure_reasons']}")
    for r in results:
        lines.extend(["", f"## Group {r['group']}, iteration {r['checkpoint_iteration']}", "",
                      f"Checkpoint: `{r['checkpoint']}`", "",
                      f"Failure reasons: `{r['exclusive_failure_reasons']}`", "",
                      f"Push events in scored episodes: {r['push_events_in_scored_episodes']}"])
    (args.eval_root / "CHECKPOINT_EVAL_REPORT.md").write_text("\n".join(lines) + "\n")
    if args.wandb:
        import wandb
        with wandb.init(
            entity="yumou0319-", project="WholeBodyTracking", mode="online",
            name="sub9_clothesstand_058_8k_12k_eval_s42",
            group="sub9_clothesstand_058_checkpoint_eval",
            dir=str(args.eval_root),
            config={"task": report["task"], "protocol": report["protocol"],
                    "seed": 42, "num_envs": 32, "episodes_per_env": args.quota,
                    "checkpoint_iterations": sorted({r["checkpoint_iteration"] for r in results})},
        ) as run:
            run.define_metric("checkpoint_iteration")
            run.define_metric("group*", step_metric="checkpoint_iteration")
            for step in sorted({r["checkpoint_iteration"] for r in results}):
                metrics = {"checkpoint_iteration": step}
                for r in results:
                    if r["checkpoint_iteration"] != step:
                        continue
                    prefix = f"group{r['group']}"
                    values = {key: r[key] for key in ("success_rate", "successes", "episodes_scored", "mean_episode_seconds", "mean_reached_frame", "push_events_in_scored_episodes")}
                    values.update(r["trajectory_errors"])
                    metrics.update({f"{prefix}/{key}": value for key, value in values.items()})
                run.log(metrics)
            (args.eval_root / "wandb_url.txt").write_text(run.url + "\n")
            print(f"W&B: {run.url}")


if __name__ == "__main__":
    main()
