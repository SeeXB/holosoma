#!/usr/bin/env python3
"""Apply the contact-union rule to this batch's saved English model responses."""

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np

from generate_remaining37_semantic_plans import COMPLETED, DATA, TASK_OBJECTS, VERSION, generate_one, write_json
from holosoma_retargeting.semantic_keyframes import local_phases as local
from holosoma_retargeting.semantic_keyframes import pipeline
from monitor_remaining37_semantic_plans import validate_output


def main():
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    run = audit / "hand_contact_regeneration_en_20260920"
    pipeline.load_env_file(Path("src/holosoma_retargeting/.env"))

    def no_new_request(*args, **kwargs):
        raise RuntimeError("This local replay must reuse saved responses, not make a new API request")

    pipeline.call_vlm = no_new_request
    rows = []
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in TASK_OBJECTS:
            if task in COMPLETED:
                continue
            bundle = DATA / "omomo/bundles" / task / "input/omomo_gt_sequence.npz"
            signals, _, fps = pipeline.load_retargeting_bundle_signals(bundle)
            count = len(next(iter(signals.values())))
            frames = np.unique(np.linspace(0, count - 1, 20).round().astype(int))
            with np.load(bundle, allow_pickle=False) as data:
                object_name = str(data["object_name"].item())
            prompt = local.phase_prompt(len(frames), False, object_name)
            source = run / "responses/omomo" / task / "segment_000"
            selected, visual = None, None
            errors = []
            if (source / "prompt.txt").read_text() != prompt:
                raise ValueError(f"{task}: saved response prompt does not match English prompt")
            for response in sorted(source.glob("attempt_*.txt"), reverse=True):
                try:
                    visual = local.validate_phases(pipeline._extract_json_value(response.read_text()), len(frames), False, require_contact_review=True)
                    actions = local.refine_actions_from_gt(local.compile_actions(visual, frames, 0), signals, fps)
                    resolved = local.execute_local_plan({"schema": local.SCHEMA, "actions": actions}, signals, fps)
                    issues = pipeline.dynamic_resolution_validation_issues(resolved, count)
                    if issues:
                        raise ValueError(issues)
                    selected = response
                    break
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(str(exc))
            if selected is None:
                rows.append({"task": task, "status": "failed", "errors": errors})
                continue
            signature = hashlib.sha256((prompt + json.dumps(frames.tolist()) + str(fps)).encode()
                                       + np.stack([signals[name] for name in sorted(signals)]).tobytes()).hexdigest()
            write_json(root / "omomo" / task / "localized_phase_cache/segment_000.json",
                       {"signature": signature, "visual_plan": visual})
            row = generate_one("omomo", task, root, run / "union_replay", 0, True)
            row["replayed_response"] = str(selected)
            checked = validate_output(root, "omomo", task)
            if checked["status"] != "validated":
                row.update(status="failed", error=checked.get("error"))
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("task", "status", "error") if k in row}), flush=True)
        write_json(run / "union_replay_summary.json", {"results": rows})
    if len(rows) != 18 or any(r["status"] != "ok" for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
