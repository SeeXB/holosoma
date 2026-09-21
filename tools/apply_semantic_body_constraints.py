#!/usr/bin/env python3
"""Apply explicit bimanual task constraints locally, without another VLM request."""

import fcntl
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from generate_remaining37_semantic_plans import DATA, VERSION, write_json
from holosoma_retargeting.semantic_keyframes.body_constraints import apply_body_part_constraints
from holosoma_retargeting.semantic_keyframes.local_phases import SCHEMA
from holosoma_retargeting.semantic_keyframes.pipeline import execute_dynamic_plan, load_retargeting_bundle_signals
from monitor_remaining37_semantic_plans import validate_output


def main():
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = audit / "body_constraint_updates" / stamp
    rules = json.loads((root / "task_body_constraints.json").read_text())
    changes = []
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for key, constraint in rules.items():
            dataset, task = key.split("/")
            file = root / key / "semantic_plan.json"
            event_file = file.with_name("semantic_plan.event_plan.json")
            result = json.loads(file.read_text())
            old_plan = json.loads(event_file.read_text())
            if old_plan.get("schema") != SCHEMA:
                raise ValueError(f"{key}: local-phase plan required")
            plan = {**old_plan, "body_part_constraints": constraint}
            plan = apply_body_part_constraints(plan)
            if plan == old_plan and validate_output(root, dataset, task)["status"] == "validated":
                continue
            bundle = DATA / "omomo/bundles" / task / "input/omomo_gt_sequence.npz" if dataset == "omomo" else root / key / "lafan_gt_sequence.npz"
            signals, _, fps = load_retargeting_bundle_signals(bundle)
            resolved = execute_dynamic_plan(plan, signals, fps)
            # This correction must change only the annotated parts, never the time boundaries or labels.
            expected = json.loads(json.dumps(result["events"]))
            for before, after in zip(expected, resolved["events"], strict=True):
                before["body_parts"] = after["body_parts"]
            if expected != resolved["events"]:
                raise ValueError(f"{key}: unexpected change outside body_parts")
            result["events"] = resolved["events"]
            result.setdefault("generation_metadata", {})["body_part_constraints"] = constraint
            for path, value in ((event_file, plan), (file, result)):
                previous = path.read_bytes()
                saved = backup / key / path.name
                saved.parent.mkdir(parents=True, exist_ok=True)
                saved.write_bytes(previous)
                write_json(path, value)
                changes.append({"path": str(path), "backup": str(saved),
                                "before_sha256": hashlib.sha256(previous).hexdigest(),
                                "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            checked = validate_output(root, dataset, task)
            if checked["status"] != "validated":
                raise ValueError(checked)
        write_json(backup / "audit.json", {"reason": "User-requested pairing for known bimanual tasks", "changes": changes})
    print(json.dumps({"updated_files": len(changes), "audit": str(backup / 'audit.json')}))


if __name__ == "__main__":
    main()
