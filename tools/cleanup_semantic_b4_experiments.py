#!/usr/bin/env python3
"""Prune superseded retarget/eval artifacts after final Semantic-B4 staging.

The default mode is a dry run.  ``--apply`` is accepted only when the staged
dataset manifest is complete (15 inputs, 13 successful RL conversions).  Every
deleted path and its pre-delete size is recorded under ``exp/data_layout``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
DATE = "2026-09-22"
BATCH_TAG = "dynamic_json_intermimic_v6_semantic_b4_active_pair_cap10_tol10mm_all15"

TASK_BUNDLES = {
    "sub03_largebox3",
    "sub10_largebox_089",
    "sub10_whitechair_118",
    "sub11_trashcan_024",
    "sub14_woodchair_004",
    "sub15_plasticbox_013",
    "sub15_suitcase_053",
    "sub15_woodchair_020",
    "sub16_largetable_013",
    "sub16_whitechair_002",
    "sub1_largetable_028",
    "sub1_plasticbox_077",
    "sub3_smalltable_013",
    "sub4_suitcase_031",
    "sub7_smallbox_023",
}

UNUSED_OMOMO_TASKS = {
    "sub11_monitor_127",
    "sub12_smalltable_032",
    "sub12_tripod_041",
    "sub16_largebox_007",
    "sub17_floorlamp_026",
    "sub9_clothesstand_058",
    "sub9_monitor_025",
    "sub9_tripod_015",
}

GOOD_RUNS = {
    "20260827_145241-b4_omni_paperdr_supported_s42-locomotion",
    "20260829_225620-b4_s1_semantic_uniform_paperdr_s42-locomotion",
    "20260829_225647-b4_s2_semantic_adaptive_paperdr_s42-locomotion",
    "20260913_181327-sub10_largebox_089_originaltraj_originalrl_s42_officialpt_default1mm_v1-locomotion",
    "20260913_181327-sub10_largebox_089_semanticb4traj_originalrl_s42_officialpt_default1mm_v1-locomotion",
    "20260913_181327-sub10_largebox_089_semanticb4traj_semanticadaptive_s42_officialpt_default1mm_v1-locomotion",
}

GOOD_RUN_FILES = {"holosoma_config.yaml", "model_29999.pt", "model_29999.onnx"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.absolute().relative_to(REPO))


def size_without_following_links(path: Path) -> int:
    if path.is_symlink():
        return path.lstat().st_size
    if path.is_file():
        return path.stat().st_size
    if not path.exists():
        return 0
    total = path.stat().st_size
    for child in path.iterdir():
        total += size_without_following_links(child)
    return total


def delete(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def children_except(directory: Path, keep: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted((path for path in directory.iterdir() if path.name not in keep), key=str)


def build_delete_targets() -> list[Path]:
    targets: list[Path] = []

    retarget_root = REPO / "exp/retargeting"
    targets.extend(
        children_except(
            retarget_root,
            {
                BATCH_TAG,
                # The latest largetable scene retains its exact historical hash
                # and contains this compatibility asset path.
                "sub1_largetable_028_officialpt_compound_20260917",
            },
        )
    )
    largetable_compat = retarget_root / "sub1_largetable_028_officialpt_compound_20260917"
    targets.extend(children_except(largetable_compat, {"assets"}))

    benchmark_root = REPO / "exp"
    targets.extend(
        sorted(
            path
            for path in benchmark_root.glob("benchmark_results*")
            if path.name != "benchmark_results_full_event_transition_truncation"
        )
    )
    final_benchmark = benchmark_root / "benchmark_results_full_event_transition_truncation"
    targets.extend(
        children_except(
            final_benchmark,
            {
                "benchmark_metadata.json",
                "final_method.json",
                "qpos_causal_diagnostics.json",
                "rl",
                "runs",
                "transition_truncation_event_metrics.csv",
                "transition_truncation_report.md",
                "transition_truncation_summary.csv",
                "videos",
            },
        )
    )
    targets.extend(
        children_except(final_benchmark / "runs", {"transition_truncated_b4"})
    )
    targets.extend(
        children_except(
            final_benchmark / "rl",
            {"transition_truncated_b4_mj_fps50_w_obj.npz"},
        )
    )

    eval_root = REPO / "exp/eval"
    targets.extend(children_except(eval_root, {"README.md", "paper_dr_robustness", "sub10_largebox_089"}))
    targets.extend(
        children_except(
            eval_root / "sub10_largebox_089",
            {"20260916_officialpt_default1mm_v1_paper_dr"},
        )
    )

    training_root = REPO / "exp/training"
    targets.extend(children_except(training_root, {"sub10_largebox_089"}))
    targets.extend(children_except(training_root / "sub10_largebox_089", {"motions"}))

    logs_root = REPO / "logs/WholeBodyTracking"
    targets.extend(children_except(logs_root, GOOD_RUNS))
    for run_name in sorted(GOOD_RUNS):
        targets.extend(children_except(logs_root / run_name, GOOD_RUN_FILES))

    semantic_exp_root = REPO / "exp/semantic_plans"
    targets.extend(children_except(semantic_exp_root, {"dynamic_json_intermimic_v6"}))

    frozen_root = (
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/frozen"
    )
    targets.extend(children_except(frozen_root, {BATCH_TAG}))

    bundles_root = REPO / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles"
    targets.extend(children_except(bundles_root, TASK_BUNDLES))
    for bundle_name in sorted(TASK_BUNDLES):
        semantic_dir = bundles_root / bundle_name / "semantic_keyframes"
        targets.extend(
            children_except(
                semantic_dir,
                {"semantic_plan.json", "semantic_plan.event_plan.json"},
            )
        )
    targets.extend(
        [
            bundles_root / "sub03_largebox3/input/omomo_gt_sequence_dynamic.npz",
            bundles_root / "sub03_largebox3/input/omomo_gt_sequence_dynamic.metadata.json",
        ]
    )

    official_root = (
        REPO / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/official_inputs"
    )
    unused_official_tasks = {
        "sub11_monitor_127",
        "sub12_smalltable_032",
        "sub12_tripod_041",
        "sub9_monitor_025",
        "sub9_tripod_015",
    }
    for task in sorted(unused_official_tasks):
        targets.extend([official_root / f"{task}.pt", official_root / ".segments" / f"{task}.bin"])
    targets.extend(
        [
            official_root / "scenes/g1_29dof_w_monitor.xml",
            official_root / "scenes/g1_29dof_w_tripod.xml",
        ]
    )

    cari4d_root = REPO / "exp/omomo_cari4d"
    for task in sorted(UNUSED_OMOMO_TASKS):
        targets.extend([cari4d_root / task, cari4d_root / f"{task}.render.log"])
    sub10_compat_plan = (
        cari4d_root
        / "sub10_largebox_089/semantic_keyframes/sub10_largebox_089_dynamic.json"
    )
    for bundle_name in sorted(TASK_BUNDLES):
        task_root = cari4d_root / bundle_name
        if not task_root.is_dir():
            continue
        for path in task_root.rglob("*"):
            if path.is_symlink() and not path.exists() and path != sub10_compat_plan:
                targets.append(path)

    training_data = REPO / "src/holosoma/holosoma/data"
    targets.extend(
        [
            training_data / "motions/retargeted",
            training_data / "motions/tasks/sub9_clothesstand_058",
            training_data / "motions/tasks/sub1_largetable_028/officialpt_compound1mm",
            training_data / "semantic/sub9_clothesstand_058",
            training_data / "experiments/sub1_largetable_028",
        ]
    )
    old_sub10 = training_data / "motions/tasks/sub10_largebox_089"
    if old_sub10.is_dir():
        targets.extend(sorted(path for path in old_sub10.iterdir() if path.is_file()))

    # Drop absent entries and nested duplicates while preserving deterministic order.
    existing = sorted({path.absolute() for path in targets if path.is_symlink() or path.exists()}, key=str)
    result: list[Path] = []
    for path in existing:
        if any(parent == prior for prior in result for parent in path.parents):
            continue
        result.append(path)
    return result


def assert_untracked(targets: list[Path]) -> None:
    if not targets:
        return
    relative_targets = [relative(path) for path in targets]
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--", *relative_targets],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    tracked = [entry for entry in proc.stdout.decode().split("\0") if entry]
    if tracked:
        raise RuntimeError(f"Cleanup refuses to delete tracked files: {tracked}")


def deduplicate_body_model(*, apply: bool) -> dict[str, Any]:
    bundles_root = REPO / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles"
    files = [
        bundles_root / bundle / "input/SMPLH_male_bodymodel_10betas.npz"
        for bundle in sorted(TASK_BUNDLES)
    ]
    missing = [relative(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing bundle body models: {missing}")
    hashes = {sha256(path) for path in files}
    if len(hashes) != 1:
        raise RuntimeError(f"Bundle body models are not identical: {hashes}")
    size = files[0].stat().st_size
    destination = (
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/body_models/omomo"
        / "SMPLH_male_bodymodel_10betas.npz"
    )
    already_linked = destination.is_file() and all(
        path.is_symlink() and path.resolve() == destination.resolve() for path in files
    )
    if apply and not already_linked:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(files[0], destination)
        if sha256(destination) not in hashes:
            raise RuntimeError(f"Central body model hash mismatch: {destination}")
        for path in files:
            path.unlink()
            path.symlink_to(os.path.relpath(destination, start=path.parent))
    return {
        "source_count": len(files),
        "sha256": next(iter(hashes)),
        "bytes_each": size,
        "estimated_bytes_saved": 0 if already_linked else (len(files) - 1) * size,
        "central_path": relative(destination),
        "bundle_links": [relative(path) for path in files],
    }


def repair_compatibility_links(*, apply: bool) -> dict[str, Any]:
    link = (
        REPO
        / "exp/omomo_cari4d/sub10_largebox_089/semantic_keyframes"
        / "sub10_largebox_089_dynamic.json"
    )
    target = (
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles"
        / "sub10_largebox_089/semantic_keyframes/semantic_plan.json"
    )
    if not target.is_file():
        raise RuntimeError(f"Missing canonical sub10 semantic plan: {target}")
    if apply:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(target, start=link.parent))
    return {
        "path": relative(link),
        "target": relative(target),
        "target_sha256": sha256(target),
        "reason": (
            "The preserved historical sub10 semantic-adaptive config uses this path. "
            "The InterMimic re-resolution summary records identical old/new event windows."
        ),
    }


def hash_preserved_files() -> list[dict[str, Any]]:
    paths = [
        REPO / "exp/retargeting" / BATCH_TAG / "AGGREGATE_REPORT.md",
        REPO / "exp/retargeting" / BATCH_TAG / "manifest.json",
        REPO / "exp/eval/paper_dr_robustness/PAPER_DR_ROBUSTNESS_REPORT.md",
        REPO / "exp/eval/paper_dr_robustness/paper_dr_robustness_results.json",
        REPO
        / "exp/eval/sub10_largebox_089/20260916_officialpt_default1mm_v1_paper_dr"
        / "FINAL_EVAL_REPORT.md",
        REPO
        / "exp/eval/sub10_largebox_089/20260916_officialpt_default1mm_v1_paper_dr"
        / "final_results.json",
        REPO / "exp/data_layout/final_semantic_b4_20260922.json",
    ]
    logs_root = REPO / "logs/WholeBodyTracking"
    for run_name in sorted(GOOD_RUNS):
        paths.extend(logs_root / run_name / name for name in sorted(GOOD_RUN_FILES))
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required preserved files are missing: {missing}")
    return [
        {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in paths
    ]


def verify_staging_manifest() -> None:
    path = REPO / "exp/data_layout/final_semantic_b4_20260922.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "tasks": 15,
        "retarget_successes": 13,
        "retarget_failures": 2,
        "rl_motions": 13,
    }
    if manifest.get("summary") != expected:
        raise RuntimeError(f"Staging manifest summary mismatch: {manifest.get('summary')}")
    for record in manifest["rl_conversions"].values():
        output = REPO / record["output"]["path"]
        if not output.is_file() or sha256(output) != record["output"]["sha256"]:
            raise RuntimeError(f"Staged RL output failed verification: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "exp/data_layout/cleanup_20260922.json",
    )
    args = parser.parse_args()

    verify_staging_manifest()
    preserved = hash_preserved_files()
    targets = build_delete_targets()
    assert_untracked(targets)
    deletion_records = [
        {"path": relative(path), "bytes": size_without_following_links(path)}
        for path in targets
    ]
    body_model = deduplicate_body_model(apply=False)
    manifest = {
        "schema_version": 1,
        "date": DATE,
        "applied": args.apply,
        "retained": preserved,
        "deleted": deletion_records,
        "deleted_bytes": sum(item["bytes"] for item in deletion_records),
        "body_model_deduplication": body_model,
        "compatibility_link": repair_compatibility_links(apply=False),
        "recoverability": (
            "Deleted generated artifacts are not recoverable from this workspace; "
            "tracked source files were explicitly excluded."
        ),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if args.apply:
        for path in targets:
            delete(path)
        manifest["body_model_deduplication"] = deduplicate_body_model(apply=True)
        manifest["compatibility_link"] = repair_compatibility_links(apply=True)
        # Re-hash the retained evidence after deletion to prove it was untouched.
        after = hash_preserved_files()
        if after != preserved:
            raise RuntimeError("A preserved file changed during cleanup")
        manifest["retained_after_cleanup"] = after
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "applied": args.apply,
                "delete_targets": len(targets),
                "deleted_gib": manifest["deleted_bytes"] / (1024**3),
                "deduplicated_gib": body_model["estimated_bytes_saved"] / (1024**3),
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
