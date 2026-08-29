#!/usr/bin/env python3
"""Run the reset-sampling sanity check without launching a simulator."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from holosoma.managers.command.semantic_transition_sampler import SemanticTransitionSampler
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-file", required=True, type=Path)
    parser.add_argument("--motion-fps", required=True, type=float)
    parser.add_argument("--motion-steps", required=True, type=int)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("exp/semantic_transition_sampling"))
    return parser.parse_args()


def _sample_original(motion_steps: int, motion_fps: float, samples: int) -> tuple[np.ndarray, np.ndarray]:
    sampler = AdaptiveTimestepsSampler(motion_steps, "cpu", int(round(motion_fps)))
    frames = sampler.sample_global_time_steps(samples).numpy()
    return frames, np.full(samples, -1, dtype=np.int64)


def _sample_semantic(
    semantic_file: Path,
    motion_steps: int,
    motion_fps: float,
    samples: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, SemanticTransitionSampler]:
    sampler = SemanticTransitionSampler(
        motion_time_step_total=motion_steps,
        num_envs=1,
        device="cpu",
        motion_fps=motion_fps,
        semantic_file=semantic_file,
        semantic_fps=None,
        sampling_mode=mode,
    )
    if mode == "semantic_adaptive":
        # Sanity check the intended direction of adaptation without claiming
        # this artificial score is a training result.
        sampler.failure_score[0] = 1.0
    frames, transition_ids, _ = sampler.sample(samples)
    return frames.numpy(), transition_ids.numpy(), sampler


def main() -> None:
    args = _parse_args()
    if args.samples < 10000:
        raise ValueError("The sanity check requires at least 10000 samples")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, tuple[np.ndarray, np.ndarray, SemanticTransitionSampler | None]] = {}
    original_frames, original_ids = _sample_original(args.motion_steps, args.motion_fps, args.samples)
    results["original_adaptive"] = (original_frames, original_ids, None)
    for mode in ("semantic_uniform", "semantic_adaptive"):
        frames, ids, sampler = _sample_semantic(
            args.semantic_file, args.motion_steps, args.motion_fps, args.samples, mode
        )
        results[mode] = (frames, ids, sampler)

    rows: list[dict[str, object]] = []
    for mode, (frames, transition_ids, sampler) in results.items():
        rows.append(
            {
                "mode": mode,
                "samples": len(frames),
                "frame_min": int(frames.min()),
                "frame_max": int(frames.max()),
                "global_uniform_fraction": 0.0 if sampler is None else float(np.mean(transition_ids < 0)),
                "transition_histogram": "" if sampler is None else ",".join(
                    str(int(value))
                    for value in np.bincount(
                        transition_ids[transition_ids >= 0], minlength=sampler.num_transitions
                    )
                ),
            }
        )
    with (args.output_dir / "semantic_sampling_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # The pure sampling sanity check has no rollout terminations, so this file
    # is intentionally all zeros.  During training the same histogram is
    # accumulated by SemanticTransitionSampler from actual early terminations.
    failure_histogram = np.zeros(args.motion_steps, dtype=np.int64)
    for _, (_, _, sampler) in results.items():
        if sampler is not None:
            failure_histogram += sampler.failure_frame_histogram.numpy()
    with (args.output_dir / "failure_frame_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("motion_frame", "num_failures"))
        writer.writerows((frame, int(count)) for frame, count in enumerate(failure_histogram))

    # Keep the transition-level attribution alongside the frame histogram.  A
    # no-rollout sanity run has zero failures; during training this same table
    # can be populated from the sampler's accumulated transition histogram.
    semantic_samplers = [sampler for _, (_, _, sampler) in results.items() if sampler is not None]
    reference_sampler = semantic_samplers[0]
    with (args.output_dir / "failure_transition_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("transition_id", "transition_name_for_log", "num_failures", "failure_rate"))
        for transition_id, transition in enumerate(reference_sampler.transitions):
            failures = sum(
                int(sampler.failure_transition_histogram[transition_id].sum().item())
                for sampler in semantic_samplers
            )
            attempts = sum(int(sampler.sample_count[transition_id].item()) for sampler in semantic_samplers)
            writer.writerow(
                (
                    transition_id,
                    f"{transition.source_event_name}->{transition.target_event_name}",
                    failures,
                    failures / attempts if attempts else 0.0,
                )
            )
    failure_figure, failure_axis = plt.subplots(figsize=(12, 4), constrained_layout=True)
    failure_axis.bar(np.arange(args.motion_steps), failure_histogram, width=1.0, color="#c0504d")
    failure_axis.set_title("Failure frame histogram (sampling sanity; no rollouts)")
    failure_axis.set_xlabel("motion frame")
    failure_axis.set_ylabel("num early terminations")
    failure_figure.savefig(args.output_dir / "failure_frame_histogram.png", dpi=160)
    plt.close(failure_figure)

    bins = np.arange(args.motion_steps + 1) - 0.5
    figure, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for row, (mode, (frames, transition_ids, sampler)) in enumerate(results.items()):
        axes[row, 0].hist(frames, bins=bins, color="#4472c4", alpha=0.85)
        axes[row, 0].set_title(f"{mode}: sampled reference frames")
        axes[row, 0].set_xlabel("motion frame")
        axes[row, 0].set_ylabel("count")
        if sampler is None:
            axes[row, 1].axis("off")
            continue
        valid = transition_ids[transition_ids >= 0]
        transition_bins = np.arange(sampler.num_transitions + 1) - 0.5
        axes[row, 1].hist(valid, bins=transition_bins, color="#70ad47", alpha=0.85)
        axes[row, 1].set_title(f"{mode}: transition IDs (global fallback omitted)")
        axes[row, 1].set_xlabel("transition id")
        axes[row, 1].set_ylabel("count")
        axes[row, 1].set_xticks(np.arange(sampler.num_transitions))
    figure.savefig(args.output_dir / "semantic_sampling_distribution.png", dpi=160)
    plt.close(figure)

    print(f"Wrote {args.output_dir / 'semantic_sampling_distribution.png'}")
    print(f"Wrote {args.output_dir / 'semantic_sampling_distribution.csv'}")
    print(f"Wrote {args.output_dir / 'failure_transition_summary.csv'}")
    for mode, (_, _, sampler) in results.items():
        if sampler is not None:
            probs = sampler.sampling_probabilities.tolist()
            print(f"{mode}: probabilities={probs}")


if __name__ == "__main__":
    main()
