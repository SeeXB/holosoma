#!/usr/bin/env python3
"""Regenerate unfinished OMOMO with explicit contact review, preserving old inputs."""

import argparse
import fcntl
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from generate_remaining37_semantic_plans import COMPLETED, DATA, TASK_OBJECTS, VERSION, generate_one, write_json
from holosoma_retargeting.semantic_keyframes.local_phases import OBJECT_CONTACT_PROMPT_VERSION
from holosoma_retargeting.semantic_keyframes.pipeline import load_env_file
from monitor_remaining37_semantic_plans import validate_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    tasks = [t for t in TASK_OBJECTS if t not in COMPLETED]
    if args.tasks and set(args.tasks) - set(tasks):
        parser.error("tasks must be unfinished OMOMO tasks")
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.run_dir / "manifest.json"
        if not manifest_path.exists():
            manifest = {"prompt_version": OBJECT_CONTACT_PROMPT_VERSION, "tasks": tasks, "backups": {}, "protected_lafan": {}}
            for task in tasks:
                directory = root / "omomo" / task
                for path in [*directory.glob("semantic_plan*.json"), *directory.glob("localized_phase_cache/*.json")]:
                    backup = args.run_dir / "before" / path.relative_to(root)
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, backup)
                    manifest["backups"][str(path)] = {"backup": str(backup), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (root / "lafan").glob("*/semantic_plan*.json"):
                manifest["protected_lafan"][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            write_json(manifest_path, manifest)
        manifest = json.loads(manifest_path.read_text())
        if manifest["prompt_version"] != OBJECT_CONTACT_PROMPT_VERSION:
            raise ValueError("use a new run directory after a prompt version change")
        load_env_file(Path("src/holosoma_retargeting/.env"))
        rows = []
        for task in tasks:
            output = root / "omomo" / task / "semantic_plan.json"
            existing = json.loads(output.read_text()) if output.exists() else {}
            fresh = existing.get("generation_metadata", {}).get("contact_prompt_version") == OBJECT_CONTACT_PROMPT_VERSION
            fresh = fresh and existing.get("generation_metadata", {}).get("contact_inclusion_policy") == "motion_parts_union_identified_contact_parts_v1"
            if args.tasks and task not in args.tasks:
                rows.append({"task": task, "status": "not_selected", "current_prompt": fresh})
                continue
            checked = validate_output(root, "omomo", task)
            if fresh and checked["status"] == "validated":
                row = {"task": task, "status": "reused_current_prompt"}
            else:
                print(json.dumps({"task": task, "status": "generating"}), flush=True)
                row = generate_one("omomo", task, root, args.run_dir / "responses", 3, True)
                if row["status"] == "ok":
                    checked = validate_output(root, "omomo", task)
                    if checked["status"] != "validated":
                        row.update(status="failed", error=checked.get("error"))
                write_json(args.run_dir / "results" / (task + ".json"), row)
            rows.append(row)
            write_json(args.run_dir / "summary.json", {"updated_at": datetime.now(timezone.utc).isoformat(), "prompt_version": OBJECT_CONTACT_PROMPT_VERSION, "results": rows})
            print(json.dumps({k: row[k] for k in ("task", "status", "error") if k in row}), flush=True)
            if "quota" in row.get("error", "").lower():
                break
        changed = [p for p, sha in manifest["protected_lafan"].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
        if changed:
            raise RuntimeError(f"LAFAN plans changed: {changed}")
        if any(r["status"] not in ("ok", "reused_current_prompt", "not_selected") for r in rows):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
