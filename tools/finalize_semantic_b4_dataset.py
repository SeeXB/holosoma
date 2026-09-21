#!/usr/bin/env python3
"""Stage the final Semantic-B4 inputs and convert successful trajectories for RL.

This is intentionally tied to the 2026-09-22 cap10/tol10mm batch.  It copies
the exact frozen inputs into the per-task OMOMO bundles, converts only runs
whose saved metrics report ``status=ok``, and writes hashes so cleanup can be
audited afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO = Path(__file__).resolve().parents[1]
RETARGET_PACKAGE = REPO / "src/holosoma_retargeting"
if str(RETARGET_PACKAGE) not in sys.path:
    sys.path.insert(0, str(RETARGET_PACKAGE))

from holosoma_retargeting.config_types.data_conversion import DataConversionConfig  # noqa: E402
from holosoma_retargeting.data_conversion.convert_data_format_mj import run_simulator  # noqa: E402


BATCH_TAG = "dynamic_json_intermimic_v6_semantic_b4_active_pair_cap10_tol10mm_all15"
RL_VERSION = "semantic_b4_active_pair_cap10_tol10mm"

TASKS = {
    "sub10_largebox_089": ("sub10_largebox_089", "largebox"),
    "sub10_whitechair_118": ("sub10_whitechair_118", "whitechair"),
    "sub11_trashcan_024": ("sub11_trashcan_024", "trashcan"),
    "sub14_woodchair_004": ("sub14_woodchair_004", "woodchair"),
    "sub15_plasticbox_013": ("sub15_plasticbox_013", "plasticbox"),
    "sub15_suitcase_053": ("sub15_suitcase_053", "suitcase"),
    "sub15_woodchair_020": ("sub15_woodchair_020", "woodchair"),
    "sub16_largetable_013": ("sub16_largetable_013", "largetable"),
    "sub16_whitechair_002": ("sub16_whitechair_002", "whitechair"),
    "sub1_largetable_028": ("sub1_largetable_028", "largetable"),
    "sub1_plasticbox_077": ("sub1_plasticbox_077", "plasticbox"),
    "sub3_largebox_003": ("sub03_largebox3", "largebox"),
    "sub3_smalltable_013": ("sub3_smalltable_013", "smalltable"),
    "sub4_suitcase_031": ("sub4_suitcase_031", "suitcase"),
    "sub7_smallbox_023": ("sub7_smallbox_023", "smallbox"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO))


def copy_exact(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"Hash mismatch after copying {source} to {destination}")


def rewrite_bundle_scene(source: Path, destination: Path) -> None:
    """Copy a scene, replacing the largetable's historical exp compatibility path."""
    text = source.read_text(encoding="utf-8")
    old = str(REPO / "exp/retargeting/sub1_largetable_028_officialpt_compound_20260917/assets")
    new = str(
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/retarget_inputs"
        / "sub1_largetable_028_officialpt_compound_20260917/assets"
    )
    text = text.replace(old, new)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def paths_for(task: str) -> dict[str, Path]:
    bundle_name, _ = TASKS[task]
    run = (
        REPO
        / "exp/retargeting"
        / BATCH_TAG
        / "omomo"
        / task
        / "semantic_b4"
    )
    bundle = (
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles"
        / bundle_name
    )
    plan_source = (
        REPO
        / "src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/frozen"
        / BATCH_TAG
        / "omomo"
        / task
        / "semantic_plan.json"
    )
    event_source = (
        REPO
        / "exp/semantic_plans/dynamic_json_intermimic_v6/retarget_plan_root/omomo"
        / task
        / "semantic_plan.event_plan.json"
    )
    semantic_bundle_source = (
        REPO
        / "exp/semantic_plans/dynamic_json_intermimic_v6"
        / task
        / "input/intermimic_semantic_bundle.npz"
    )
    return {
        "run": run,
        "bundle": bundle,
        "plan_source": plan_source,
        "event_source": event_source,
        "semantic_bundle_source": semantic_bundle_source,
    }


def stage_task(task: str) -> dict[str, Any]:
    paths = paths_for(task)
    run = paths["run"]
    bundle = paths["bundle"]
    job_path = run / "job.json"
    metrics_path = run / "metrics.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    pt_sources = [Path(name) for name in job["sources"] if name.endswith(".pt")]
    if len(pt_sources) != 1:
        raise RuntimeError(f"Expected one PT source for {task}, found {pt_sources}")
    scene_source = Path(job["scene"])

    staged = {
        "intermimic_official_pt": bundle / "input/intermimic_official.pt",
        "retarget_scene": bundle / "input/retarget_scene.xml",
        "intermimic_semantic_bundle": bundle / "input/intermimic_semantic_bundle.npz",
        "semantic_plan": bundle / "semantic_keyframes/semantic_plan.json",
        "event_plan": bundle / "semantic_keyframes/semantic_plan.event_plan.json",
    }
    copy_exact(pt_sources[0], staged["intermimic_official_pt"])
    rewrite_bundle_scene(scene_source, staged["retarget_scene"])
    copy_exact(paths["semantic_bundle_source"], staged["intermimic_semantic_bundle"])
    copy_exact(paths["plan_source"], staged["semantic_plan"])
    copy_exact(paths["event_source"], staged["event_plan"])

    raw_candidates = sorted(run.glob("*_original.npz"))
    status = metrics.get("status")
    if status == "ok" and len(raw_candidates) != 1:
        raise RuntimeError(f"Successful task {task} must have exactly one result NPZ")
    if status != "ok" and raw_candidates:
        raise RuntimeError(f"Failed task {task} unexpectedly has a result NPZ")

    provenance = {
        "task": task,
        "bundle_name": bundle.name,
        "batch": BATCH_TAG,
        "method": job.get("method"),
        "status": status,
        "retarget_parameters": {
            "semantic_weights": job.get("semantic_weights"),
            "geometry_projection": job.get("geometry_projection"),
            "active_pair_nonpenetration_refinement": job.get(
                "active_pair_nonpenetration_refinement"
            ),
            "active_pair_max_iterations": job.get("active_pair_max_iterations"),
            "active_pair_acceptance_tolerance": job.get(
                "active_pair_acceptance_tolerance"
            ),
            "active_pair_prediction_margin": job.get("active_pair_prediction_margin"),
        },
        "source_job": relative(job_path),
        "source_metrics": relative(metrics_path),
        "source_scene": relative(scene_source),
        "source_scene_sha256": sha256(scene_source),
        "files": {
            name: {"path": relative(path), "sha256": sha256(path)}
            for name, path in staged.items()
        },
        "retarget_result": (
            {
                "path": relative(raw_candidates[0]),
                "sha256": sha256(raw_candidates[0]),
            }
            if raw_candidates
            else None
        ),
        "failure": metrics.get("error") if status != "ok" else None,
    }
    provenance_path = bundle / "RETARGET_INPUTS.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    provenance["provenance"] = relative(provenance_path)
    provenance["provenance_sha256"] = sha256(provenance_path)
    return provenance


def convert_task(task: str, staged: dict[str, Any]) -> dict[str, Any] | None:
    if staged["status"] != "ok":
        return None

    _, object_name = TASKS[task]
    source = REPO / staged["retarget_result"]["path"]
    scene = REPO / staged["files"]["retarget_scene"]["path"]
    output_dir = REPO / "src/holosoma/holosoma/data/motions/tasks" / task / RL_VERSION
    output = output_dir / f"{task}_semantic_b4_mj_w_obj.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_simulator(
        DataConversionConfig(
            input_file=str(source),
            object_name=object_name,
            scene_xml_file=str(scene),
            input_fps=30,
            output_fps=50,
            has_dynamic_object=True,
            output_name=str(output),
            once=True,
            headless=True,
        )
    )

    with np.load(output, allow_pickle=False) as payload:
        required = {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
            "joint_names",
            "body_names",
        }
        missing = sorted(required.difference(payload.files))
        if missing:
            raise RuntimeError(f"Converted {task} motion is missing keys: {missing}")
        frames = int(payload["joint_pos"].shape[0])
        fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
        finite_keys = sorted(required.difference({"joint_names", "body_names"}))
        non_finite = [name for name in finite_keys if not np.isfinite(payload[name]).all()]
        if non_finite:
            raise RuntimeError(f"Converted {task} motion has non-finite arrays: {non_finite}")
        if fps != 50.0 or frames < 2:
            raise RuntimeError(f"Invalid converted motion for {task}: frames={frames}, fps={fps}")
        shapes = {name: list(payload[name].shape) for name in sorted(payload.files)}

    conversion = {
        "task": task,
        "version": RL_VERSION,
        "source_retarget_result": staged["retarget_result"],
        "scene": staged["files"]["retarget_scene"],
        "output": {
            "path": relative(output),
            "sha256": sha256(output),
            "bytes": output.stat().st_size,
            "frames": frames,
            "fps": fps,
            "shapes": shapes,
        },
        "converter": "src/holosoma_retargeting/holosoma_retargeting/data_conversion/convert_data_format_mj.py",
        "parameters": {
            "data_format": "smplh",
            "object_name": object_name,
            "has_dynamic_object": True,
            "input_fps": 30,
            "output_fps": 50,
            "headless": True,
        },
    }
    provenance_path = output_dir / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(conversion, indent=2) + "\n", encoding="utf-8")
    conversion["provenance"] = relative(provenance_path)
    conversion["provenance_sha256"] = sha256(provenance_path)
    return conversion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--convert", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO / "exp/data_layout/final_semantic_b4_20260922.json",
    )
    args = parser.parse_args()
    if not args.stage:
        parser.error("--stage is required; staging is always verified before conversion")

    staged = {task: stage_task(task) for task in TASKS}
    converted: dict[str, dict[str, Any]] = {}
    if args.convert:
        for task, task_stage in staged.items():
            record = convert_task(task, task_stage)
            if record is not None:
                converted[task] = record

    manifest = {
        "schema_version": 1,
        "date": "2026-09-22",
        "batch": BATCH_TAG,
        "rl_version": RL_VERSION,
        "summary": {
            "tasks": len(staged),
            "retarget_successes": sum(item["status"] == "ok" for item in staged.values()),
            "retarget_failures": sum(item["status"] != "ok" for item in staged.values()),
            "rl_motions": len(converted),
        },
        "tasks": staged,
        "rl_conversions": converted,
        "preserved_downstream_rl": {
            "sub3_largebox_003": {
                "report": "exp/eval/paper_dr_robustness/PAPER_DR_ROBUSTNESS_REPORT.md",
                "result": "exp/eval/paper_dr_robustness/paper_dr_robustness_results.json",
            },
            "sub10_largebox_089": {
                "report": "exp/eval/sub10_largebox_089/20260916_officialpt_default1mm_v1_paper_dr/FINAL_EVAL_REPORT.md",
                "result": "exp/eval/sub10_largebox_089/20260916_officialpt_default1mm_v1_paper_dr/final_results.json",
            },
        },
        "plasticbox_downstream_rl_search": {
            "found": False,
            "note": (
                "No plasticbox training run, checkpoint, or formal RL evaluation was found. "
                "Both current plasticbox retarget trajectories are preserved and converted."
            ),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))
    print(f"manifest={args.manifest}")


if __name__ == "__main__":
    main()
