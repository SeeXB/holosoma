"""Analyze Paper-DR robustness recordings using a fixed first-10 quota.

Usage::

    python scripts/analyze_paper_dr_robustness.py \
      --s1 exp/eval/paper_dr_robustness/s1/rollout_all_envs.npz \
      --s2 exp/eval/paper_dr_robustness/s2/rollout_all_envs.npz \
      --output-dir exp/eval/paper_dr_robustness

An episode is successful only when the environment reaches the clip timeout
without any bad-tracking criterion set at its terminal pre-reset snapshot.
Exactly the first ten completed episodes of every environment are scored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REASON_KEYS = (
    "bad_ref_pos",
    "bad_ref_ori",
    "bad_motion_body_pos",
    "bad_object_pos",
    "bad_object_ori",
)
DISPLAY_PRIORITY = (
    "bad_object_pos",
    "bad_object_ori",
    "bad_motion_body_pos",
    "bad_ref_ori",
    "bad_ref_pos",
)
PHASE_BOUNDARIES = (50, 110, 195, 270, 324)


def _metadata(z: Any) -> dict[str, Any]:
    raw = z.get("_metadata_json")
    if raw is None:
        return {}
    value = raw.item() if hasattr(raw, "item") else raw
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}


def episode_rows(path: Path, quota: int = 10) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as z:
        if "done" not in z or z["done"].ndim != 2:
            raise RuntimeError(f"{path} is not an all-environment recording")
        num_envs = int(z["done"].shape[1])
        md = _metadata(z)
        for env_id in range(num_envs):
            terminal_steps = np.flatnonzero(z["done"][:, env_id].astype(bool))
            if len(terminal_steps) < quota:
                raise RuntimeError(f"{path}: env {env_id} has only {len(terminal_steps)} completed episodes")
            for episode_id, terminal_step in enumerate(terminal_steps[:quota], start=1):
                terminal_flags = tuple(k for k in REASON_KEYS if bool(z[k][terminal_step, env_id]))
                timeout = bool(z["timeout"][terminal_step, env_id])
                # In WBT eval the command manager resets its reference as soon
                # as the final motion frame is reached.  The environment then
                # raises its timeout one control tick later; the pre-reset
                # diagnostic at that tick can compare a freshly reset
                # reference against a stale PhysX object state.  For a timeout
                # episode, use the immediately preceding (still-on-clip)
                # tracking flags for the success decision and retain the raw
                # terminal flags for auditability.
                previous_flags = ()
                if timeout and terminal_step > 0:
                    previous_flags = tuple(k for k in REASON_KEYS if bool(z[k][terminal_step - 1, env_id]))
                flags = previous_flags if timeout else terminal_flags
                success = timeout and not flags
                if timeout and terminal_flags and not flags:
                    classification = "timeout_with_post_reset_stale_tracking_flags"
                elif timeout and flags:
                    classification = "timeout_with_tracking_flag"
                elif not timeout and not flags:
                    classification = "unclassified_non_timeout"
                else:
                    classification = "success" if success else "tracking_failure"
                motion_step = int(z["motion_step"][terminal_step, env_id])
                # The command sampler can reset ``motion_step`` in the same
                # control tick in which the clip timeout is raised.  A
                # timeout-with-tracking-flag therefore still reached the clip
                # horizon; use the timeout signal rather than the reset
                # timestep for phase/survival statistics.
                reached = 324 if timeout else motion_step
                previous_terminal = -1 if episode_id == 1 else int(terminal_steps[episode_id - 2])
                rows.append(
                    {
                        "env": env_id,
                        "episode": episode_id,
                        "terminal_step": int(terminal_step),
                        "start_step": previous_terminal + 1,
                        "timeout": timeout,
                        "success": success,
                        "flags": flags,
                        "terminal_flags": terminal_flags,
                        "previous_flags": previous_flags,
                        "classification": classification,
                        "motion_step": motion_step,
                        "reached": reached,
                    }
                )
        md.update(
            {
                "num_envs": num_envs,
                "rollout_envs": num_envs,
                "episodes_per_env_scored": quota,
                "eval_protocol": md.get("eval_protocol", "paper_dr_robustness_v1"),
                "paper_push_enabled_during_eval": md.get("paper_push_enabled_during_eval", True),
                "paper_shape_scale_implemented": md.get("paper_shape_scale_implemented", False),
                "termination_object_pos_threshold_m": md.get("termination_object_pos_threshold_m", 1.0),
                "termination_object_ori_threshold_rad": md.get(
                    "termination_object_ori_threshold_rad", 0.7853981633974483
                ),
            }
        )
    return rows, md


def _clustered_ci(rows: list[dict[str, Any]], num_envs: int) -> tuple[float, float]:
    rates = np.asarray(
        [np.mean([r["success"] for r in rows if r["env"] == env]) for env in range(num_envs)],
        dtype=float,
    )
    if num_envs < 2:
        return float(rates.mean()), float(rates.mean())
    half_width = 1.96 * rates.std(ddof=1) / np.sqrt(num_envs)
    return float(rates.mean() - half_width), float(rates.mean() + half_width)


def _phase(reached: int) -> str:
    if reached < 50:
        return "pre_contact_<50"
    if reached < 110:
        return "contact_to_lift_50_109"
    if reached < 195:
        return "carry_110_194"
    if reached < 270:
        return "place_195_269"
    return "release_270_323"


def _trajectory_errors(path: Path, rows: list[dict[str, Any]]) -> dict[str, float | None]:
    with np.load(path, allow_pickle=False) as z:
        required = {
            "object_pos_error_m",
            "object_ori_error_rad",
            "pre_actual_tracked_body_pos_w",
            "pre_reference_tracked_body_pos_w",
        }
        missing = sorted(required.difference(z.files))
        if missing:
            return {"object_pos_rmse_m": None, "object_ori_rmse_deg": None, "tracked_body_pos_rmse_m": None}

        object_pos_sq: list[np.ndarray] = []
        object_ori_sq: list[np.ndarray] = []
        body_sq: list[np.ndarray] = []
        for row in rows:
            start = int(row["start_step"])
            end = int(row["terminal_step"]) + 1
            env = int(row["env"])
            # Exclude the one post-clip command-reset sample from trajectory
            # errors.  The valid clip samples are motion frames 1..324.
            if bool(row["timeout"]) and int(row["motion_step"]) == 0 and end > start:
                end -= 1
            object_pos_sq.append(np.asarray(z["object_pos_error_m"][start:end, env], dtype=float) ** 2)
            object_ori_sq.append(np.asarray(z["object_ori_error_rad"][start:end, env], dtype=float) ** 2)
            delta = np.asarray(
                z["pre_actual_tracked_body_pos_w"][start:end, env]
                - z["pre_reference_tracked_body_pos_w"][start:end, env],
                dtype=float,
            )
            body_sq.append(np.sum(delta * delta, axis=-1).reshape(-1))
        return {
            "object_pos_rmse_m": float(np.sqrt(np.concatenate(object_pos_sq).mean())),
            "object_ori_rmse_deg": float(np.degrees(np.sqrt(np.concatenate(object_ori_sq).mean()))),
            "tracked_body_pos_rmse_m": float(np.sqrt(np.concatenate(body_sq).mean())),
        }


def summarize(label: str, path: Path, quota: int = 10) -> dict[str, Any]:
    rows, metadata = episode_rows(path, quota)
    num_envs = int(metadata["num_envs"])
    total = len(rows)
    successes = sum(bool(r["success"]) for r in rows)
    ci_low, ci_high = _clustered_ci(rows, num_envs)
    failures = [r for r in rows if not r["success"]]
    raw_flags = Counter(k for r in failures for k in r["flags"])
    stale_terminal_flags = Counter(k for r in rows for k in r["terminal_flags"] if k not in r["flags"])
    exclusive = Counter(next((key for key in DISPLAY_PRIORITY if key in r["flags"]), r["classification"]) for r in failures)
    combinations = Counter("+".join(r["flags"]) or r["classification"] for r in failures)
    result: dict[str, Any] = {
        "label": label,
        "recording": str(path),
        "num_envs": num_envs,
        "episodes_per_env": quota,
        "episodes_scored": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total,
        "clustered_95ci": [ci_low, ci_high],
        "success_by_episode": [sum(bool(r["success"]) for r in rows if r["episode"] == ep) for ep in range(1, quota + 1)],
        "success_by_env": [sum(bool(r["success"]) for r in rows if r["env"] == env) for env in range(num_envs)],
        "mean_reached_frame": float(np.mean([r["reached"] for r in rows])),
        "median_reached_frame": float(np.median([r["reached"] for r in rows])),
        "exclusive_failure_reasons": dict(exclusive),
        "raw_failure_flags": dict(raw_flags),
        "stale_timeout_terminal_flags": dict(stale_terminal_flags),
        "failure_flag_combinations": dict(combinations),
        "failure_phases": dict(Counter(_phase(int(r["reached"])) for r in failures)),
        "survival_at_frame": {
            str(frame): sum(int(r["reached"]) >= frame for r in rows) / total for frame in PHASE_BOUNDARIES
        },
        "trajectory_errors": _trajectory_errors(path, rows),
        "metadata": metadata,
    }
    # The cumulative counter is sampled in the recording, so the final row for
    # each env is its total number of applied pushes during the rollout.
    with np.load(path, allow_pickle=False) as z:
        if "push_count" in z:
            result["push_events_by_env"] = [int(x) for x in np.max(z["push_count"], axis=0)]
            result["push_events_total"] = int(sum(result["push_events_by_env"]))
    return result


def _markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper-DR robustness evaluation",
        "",
        "Protocol: fixed B4 reference, seed 42, 32 parallel environments, first 10 completed episodes per environment (320 scored episodes), 324 valid motion frames per clip. Object termination thresholds are 1.0 m and 45 degrees. Paper push perturbations are enabled during evaluation; object shape ±10% remains unsupported by the current simulator setup. Success is a timeout with no tracking flag on the final valid motion frame; a one-tick post-clip command-reset mismatch is reported separately.",
        "",
        "| model | SR | clustered 95% CI | successes | mean reached frame | object pos RMSE | object ori RMSE | body pos RMSE | pushes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        ci = result["clustered_95ci"]
        errors = result["trajectory_errors"]
        lines.append(
            f"| {result['label']} | {100 * result['success_rate']:.2f}% | {100 * ci[0]:.2f}%–{100 * ci[1]:.2f}% | {result['successes']}/{result['episodes_scored']} | {result['mean_reached_frame']:.1f} | {errors['object_pos_rmse_m']:.4f} m | {errors['object_ori_rmse_deg']:.2f}° | {errors['tracked_body_pos_rmse_m']:.4f} m | {result.get('push_events_total', 'n/a')} |"
        )
    for result in results:
        lines.extend(
            [
                "",
                f"### {result['label']}",
                "",
                f"- Success by episode: `{result['success_by_episode']}`",
                f"- Exclusive failure reasons: `{result['exclusive_failure_reasons']}`",
                f"- Raw failure flags: `{result['raw_failure_flags']}`",
                f"- Timeout/post-reset flags excluded from failure: `{result['stale_timeout_terminal_flags']}`",
                f"- Failure phases: `{result['failure_phases']}`",
                f"- Survival at frames: `{result['survival_at_frame']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", type=Path)
    parser.add_argument("--s2", type=Path)
    parser.add_argument("--omni", type=Path, help="A single Omni baseline recording")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=10)
    args = parser.parse_args()

    inputs = []
    if args.s1 is not None:
        inputs.append(("S1 semantic_uniform", args.s1))
    if args.s2 is not None:
        inputs.append(("S2 semantic_adaptive", args.s2))
    if args.omni is not None:
        inputs.append(("Omni baseline", args.omni))
    if not inputs:
        parser.error("provide at least one of --s1, --s2, or --omni")
    results = [summarize(label, path, args.quota) for label, path in inputs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paper_dr_robustness_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output_dir / "PAPER_DR_ROBUSTNESS_REPORT.md").write_text(_markdown(results))
    for result in results:
        print(
            f"{result['label']}: SR={100 * result['success_rate']:.2f}% "
            f"({result['successes']}/{result['episodes_scored']}), "
            f"failures={result['exclusive_failure_reasons']}, "
            f"pushes={result.get('push_events_total', 'n/a')}"
        )


if __name__ == "__main__":
    main()
