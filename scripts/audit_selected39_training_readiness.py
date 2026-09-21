"""Audit selected OMOMO/LAFAN inputs and semantics without modifying them."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src/holosoma_retargeting"), str(ROOT / "src/holosoma")]
from holosoma_retargeting.semantic_keyframes.pipeline import (
    dynamic_plan_validation_issues,
    dynamic_resolution_validation_issues,
    execute_dynamic_plan,
    load_retargeting_bundle_signals,
    validate_semantic_keyframe_json,
)
from holosoma_retargeting.src.utils import load_intermimic_data

# Load this standalone sampler without importing simulation-manager packages.
spec = importlib.util.spec_from_file_location(
    "readiness_semantic_sampler",
    ROOT / "src/holosoma/holosoma/managers/command/semantic_transition_sampler.py",
)
sampler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sampler
spec.loader.exec_module(sampler)


def npz_status(path: Path) -> dict:
    result = {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
    if not path.is_file():
        return result
    try:
        with np.load(path, allow_pickle=False) as data:
            result["keys"] = data.files
            result["finite"] = all(
                np.isfinite(data[key]).all()
                for key in data.files
                if data[key].dtype.kind in "fci"
            )
            result["shapes"] = {key: list(data[key].shape) for key in data.files}
        result["valid"] = bool(result["finite"])
    except Exception as exc:
        result["valid"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def semantic_status(path: Path, bundle: Path, frame_count: int) -> dict:
    result = {"path": str(path.relative_to(ROOT)), "issues": []}
    issues = result["issues"]
    try:
        payload = json.loads(path.read_text())
        validate_semantic_keyframe_json(payload)
        issues.extend(dynamic_resolution_validation_issues(payload, frame_count))
        events = [(e["event"], w["start_frame"], w["trigger_frame"], w["end_frame"])
                  for e in payload["events"] for w in e["windows"]]
        result["events"] = events
        for name, start, trigger, end in events:
            if not 0 <= start <= trigger <= end < frame_count:
                issues.append({"message": f"{name}: window outside input length {frame_count}"})
        event_plan = path.with_name(path.stem + ".event_plan.json")
        plan = json.loads(event_plan.read_text())
        issues.extend(dynamic_plan_validation_issues(plan))
        signals, _, fps = load_retargeting_bundle_signals(bundle)
        signal_frames = len(next(iter(signals.values())))
        result["signal_frames"] = signal_frames
        if signal_frames != frame_count:
            issues.append({"message": f"signal/input length mismatch: {signal_frames}/{frame_count}"})
        if payload.get("fps") != fps:
            issues.append({"message": f"semantic/signal FPS mismatch: {payload.get('fps')}/{fps}"})
        resolved = execute_dynamic_plan(plan, signals, fps=fps)
        issues.extend(dynamic_resolution_validation_issues(resolved, signal_frames))
        fresh = [(e["event"], w["start_frame"], w["trigger_frame"], w["end_frame"])
                 for e in resolved["events"] for w in e["windows"]]
        if events != fresh:
            issues.append({"message": "saved event windows differ from re-executed event plan"})
        # Match convert_data_format_mj.MotionLoader's half-open resampling grid.
        # Final verification must use the actual converted motion before launch.
        motion_frames = len(torch.arange(0, (frame_count - 1) / fps, 1 / 50, dtype=torch.float32))
        transitions = sampler.load_semantic_transitions(
            path, motion_fps=50.0, motion_time_step_total=motion_frames
        )
        result["rl_transition_count"] = len(transitions)
        result["predicted_rl_frames"] = motion_frames
    except Exception as exc:
        issues.append({"message": f"{type(exc).__name__}: {exc}"})
    result["valid"] = not issues
    return result


def main() -> None:
    selected = json.loads((ROOT / "exp/omomo_cari4d/selected20_semantic_summary.json").read_text())["tasks"]
    official = ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/official_inputs"
    retarget = ROOT / "third_party/holosoma_upstream_audit/original_selected20_default/runs"
    rows = []
    for task in selected:
        row = {"task": task}
        pt = official / f"{task}.pt"
        row["input"] = {"path": str(pt.relative_to(ROOT)), "exists": pt.is_file()}
        try:
            tensor = torch.load(pt, map_location="cpu", weights_only=False)
            human, obj = load_intermimic_data(str(pt))
            row["input"].update(shape=list(tensor.shape), frames=len(human), valid=bool(torch.isfinite(tensor).all()),
                                quaternion_max_norm_error=float(np.max(np.abs(np.linalg.norm(obj[:, :4], axis=1)-1))))
            bundle_root = ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/bundles" / ("sub03_largebox3" if task == "sub3_largebox_003" else task)
            plans = sorted((bundle_root / "semantic_keyframes").glob(f"{task}_dynamic*.json"))
            row["semantic_candidates"] = [semantic_status(p, bundle_root / "input/omomo_gt_sequence.npz", len(human))
                                          for p in plans if not p.name.endswith(".event_plan.json")]
        except Exception as exc:
            row["input"].update(valid=False, error=f"{type(exc).__name__}: {exc}")
            row["semantic_candidates"] = []
        row["semantic_ready"] = any(p["valid"] for p in row["semantic_candidates"])
        row["retarget"] = {}
        for method in ("original", "semantic_b4"):
            p = retarget / method / task / f"{task}_{method}.npz"
            fresh = ROOT / "exp/retargeting/sub1_largetable_028_officialpt_default1mm_20260917/runs" / method / task / f"{task}_{method}.npz"
            if (fresh.parent / "retarget.log").is_file():
                p = fresh
            row["retarget"][method] = npz_status(p)
            log = p.parent / "retarget.log"
            if not p.exists() and log.exists():
                lines = log.read_text(errors="replace").splitlines()
                row["retarget"][method]["log_last_line"] = lines[-1] if lines else "empty log"
        row["rl_motions"] = [str(p.relative_to(ROOT)) for p in (ROOT / "src/holosoma/holosoma/data/motions/tasks" / task).rglob("*.npz")]
        rows.append(row)
        print(task, "input", row["input"].get("valid"), "semantic", row["semantic_ready"],
              "retarget", {k: v.get("valid", False) for k,v in row["retarget"].items()}, flush=True)

    manifest = json.loads((ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/retarget_inputs/lafan_batch/input/manifest.json").read_text())
    lafan = []
    for task, info in manifest["tasks"].items():
        row = {"task": task, "stride": manifest["stride"], "semantic_plan": "not_defined_in_existing_protocol"}
        for key, p in [("source", Path(manifest["source"]) / f"{task}.npy"),
                       ("retarget_input", ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/retarget_inputs/lafan_batch/input" / f"{task}.npy")]:
            try:
                x = np.load(p, allow_pickle=False)
                row[key] = {"path": str(p), "shape": list(x.shape), "finite": bool(np.isfinite(x).all())}
                expected = info["source_frames"] if key == "source" else info["frames"]
                row[key]["valid"] = row[key]["finite"] and len(x) == expected
            except Exception as exc:
                row[key] = {"path": str(p), "valid": False, "error": str(exc)}
        row["retarget"] = {method: npz_status(ROOT / "exp/retargeting/lafan_batch/runs" / method / task / f"{task}.npz")
                           for method in ("original", "uniform")}
        lafan.append(row)
    output = ROOT / "exp/training/readiness_20260917"
    output.mkdir(parents=True, exist_ok=True)
    report = {"checked_at": datetime.now().astimezone().isoformat(), "omomo": rows, "lafan": lafan,
              "notes": ["Input validation is structural, not a semantic-quality or physical-feasibility guarantee.",
                        "Existing retarget arrays are checked for readability and finiteness; command provenance must be checked before reuse.",
                        "LAFAN currently uses native NPY, not InterMimic object-interaction PT; old protocol is Original vs Uniform-2."]}
    (output / "readiness.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = [
        "# Selected 39-task training readiness audit", "", f"Checked: {report['checked_at']}", "",
        "OMOMO uses the official InterMimic PT inputs under `src/holosoma_retargeting/holosoma_retargeting/demo_data/omomo/official_inputs`.",
        "Input checks cover readability, finite values and unit object quaternions. Semantic checks cover schema,",
        "resolved timeline, agreement with re-executing the event plan, source frame count/FPS and RL trigger mapping.",
        "Passing these checks does not establish visual semantic correctness or physical feasibility.", "",
        "`Re-resolve` means the existing windows differ from the current plan execution; `Invalid` means other validation failures.",
        "Retarget columns below describe the audited official-input output directories, not older OMOMO conversions.", "",
        "| # | OMOMO task | Input PT | Semantic plan | Original output | B4 output | RL motion files found |",
        "|---|---|---|---|---|---|---:|",
    ]
    for index, row in enumerate(rows, 1):
        issues = [i["message"] for p in row["semantic_candidates"] for i in p["issues"]]
        state = "Pass" if row["semantic_ready"] else (
            "Re-resolve" if issues and all(i == "saved event windows differ from re-executed event plan" for i in issues) else "Invalid")
        states = ["Present/finite" if row["retarget"][m].get("valid") else "Missing" for m in ("original", "semantic_b4")]
        lines.append(f"| {index} | {row['task']} | {'Pass' if row['input'].get('valid') else 'Fail'} | {state} | {states[0]} | {states[1]} | {len(row['rl_motions'])} |")
    lines.extend(["", "## LAFAN", "",
                  "Existing protocol: native LAFAN NPY input, stride 20 for the old retarget benchmark, Original vs Uniform-2.",
                  "No per-task semantic B4/semantic-adaptive three-group protocol is established by these artifacts.",
                  "The old downsampled benchmark outputs should not be treated as ready full-rate RL references.", "",
                  "| # | LAFAN task | Source NPY | Downsampled input | Original output | Uniform-2 output |",
                  "|---|---|---|---|---|---|"])
    for index, row in enumerate(lafan, 1):
        flags = [row['source']['valid'], row['retarget_input']['valid'],
                 row['retarget']['original'].get('valid'), row['retarget']['uniform'].get('valid')]
        lines.append(f"| {index} | {row['task']} | " + " | ".join("Pass" if f else "Fail" for f in flags) + " |")
    lines.extend(["", "## Next task", "",
                  "Following the stored OMOMO list after sub10_largebox_089, the next task is sub1_largetable_028.",
                  "Its input and semantic plan pass this audit, but the earlier official/default-1-mm Original",
                  "retarget log ends with `CVXPY solve failed: infeasible`. No RL-ready reference pair was found.",
                  "Fresh retry artifacts, if present: `exp/retargeting/sub1_largetable_028_officialpt_default1mm_20260917/`.",
                  "See readiness.json for exact file paths, shapes, failure messages and candidate-plan details.", ""])
    (output / "READINESS.md").write_text("\n".join(lines))
    print("REPORT", output / "readiness.json")


if __name__ == "__main__":
    main()
