#!/usr/bin/env python3
"""Report the English OMOMO contact regeneration, including unresolved visual issues."""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from generate_remaining37_semantic_plans import COMPLETED, DATA, TASK_OBJECTS, VERSION, write_json
from holosoma_retargeting.semantic_keyframes.local_phases import OBJECT_CONTACT_PROMPT_VERSION
from monitor_remaining37_semantic_plans import validate_output


def main():
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    run = audit / "latest_sources"
    manifest = json.loads((run / "manifest.json").read_text())
    rows = []
    for task in TASK_OBJECTS:
        if task in COMPLETED:
            continue
        file = root / "omomo" / task / "semantic_plan.json"
        current = json.loads(file.read_text())
        checked = validate_output(root, "omomo", task)
        fresh = current.get("generation_metadata", {}).get("contact_prompt_version") == OBJECT_CONTACT_PROMPT_VERSION
        fresh = fresh and current.get("generation_metadata", {}).get("contact_inclusion_policy") == "motion_parts_union_identified_contact_parts_v1"
        parts = sorted({p for e in current["events"] for p in e["body_parts"]})
        reviews = [e.get("contact_review", {}) for e in current["events"]]
        row = {"task": task, "current_prompt": fresh, "status": checked["status"],
               "hand_parts": [p for p in parts if p.endswith("_hand")],
               "event_count": len(current["events"]),
               "uncertain_contact_phases": sum(any(r.get(k) == "uncertain" for k in ("hand", "wrist", "forearm")) for r in reviews),
               "model_hand_contact_sides": dict(Counter(r.get("hand", "unreviewed") for r in reviews)),
               "body_part_constraints": current.get("generation_metadata", {}).get("body_part_constraints"),
               "contact_refinement": current.get("generation_metadata", {}).get("contact_refinement"),
               "output": str(file)}
        rows.append(row)
    count = sum(r["current_prompt"] and r["status"] == "validated" for r in rows)
    changed = [p for p, sha in manifest["protected_lafan"].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "english_prompt_validated": count,
               "total": len(rows), "protected_lafan_files": len(manifest["protected_lafan"]),
               "changed_lafan_files": changed, "visual_review": "pending", "new_training_started": False, "results": rows}
    write_json(audit / "hand_contact_review.json", payload)
    lines = ["# English prompt: OMOMO hand contact review", "", f"Updated: {payload['updated_at']}", "",
             f"**{count}/{len(rows)} regenerated with the English prompt and locally validated.**",
             f"LAFAN: {len(manifest['protected_lafan'])} plan files protected, {len(changed)} changed. No new training started.", "",
             "Prompt: `local_phases.py:phase_prompt`; prompt version: `" + OBJECT_CONTACT_PROMPT_VERSION + "`.",
             "The per-phase contact_review distinguishes hand, wrist and forearm; it is model evidence, not verified contact ground truth.",
             "body_parts includes action-important parts plus contact parts. Existing reward logic still reads their union; this update does not change rewards or forearm-to-link mapping.",
             "Known bimanual constraints add the opposite upper-limb labels; raw model contact_review is preserved and may still describe only one side.",
             "The model may conflate wrist/forearm contact with hand contact despite the prompt, and its action phases can be inaccurate. Visual review remains pending.", "",
             "Only current plans and their accepted model sources are retained. Sources: `latest_sources/`; removal inventory: `cleanup.json`.", "",
             "Tripod contact refinement responses: `latest_sources/omomo/<task>/contact_review/`. The canonical segment_000.json cache contains the final refined annotations; obsolete caches were removed.", "",
             "| Task | English/current | Local validation | Hand parts | Phases with uncertainty |",
             "|---|---|---|---|---:|"]
    for row in rows:
        target = os.path.relpath(Path(row["output"]).resolve(), audit.resolve())
        lines.append(f"| [{row['task']}]({target}) | {row['current_prompt']} | {row['status']} | {', '.join(row['hand_parts']) or 'none'} | {row['uncertain_contact_phases']} |")
    (audit / "HAND_CONTACT_REVIEW.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"english_prompt_validated": count, "total": len(rows), "changed_lafan_files": changed}))


if __name__ == "__main__":
    main()
