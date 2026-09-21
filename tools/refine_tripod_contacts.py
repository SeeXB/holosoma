#!/usr/bin/env python3
"""Focused contact-only model pass for tripod phases; retain existing temporal rules."""

import base64
import fcntl
import json
from pathlib import Path

import numpy as np

from generate_remaining37_semantic_plans import DATA, VERSION, write_json
from holosoma_retargeting.semantic_keyframes import local_phases as local
from holosoma_retargeting.semantic_keyframes import pipeline
from monitor_remaining37_semantic_plans import validate_output


def main():
    root = DATA / "semantic_keyframes" / VERSION
    audit = Path("exp/semantic_plans") / VERSION
    run = audit / "hand_contact_regeneration_en_20260920_tripod_contact_only"
    pipeline.load_env_file(Path("src/holosoma_retargeting/.env"))
    with (audit / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task in ("sub12_tripod_041", "sub9_tripod_015"):
            directory = root / "omomo" / task
            file = directory / "semantic_plan.json"
            event_file = directory / "semantic_plan.event_plan.json"
            result, plan = json.loads(file.read_text()), json.loads(event_file.read_text())
            write_json(run / task / "before.semantic_plan.json", result)
            write_json(run / task / "before.event_plan.json", plan)
            all_actions = []
            for segment in plan["segments"]:
                for index, phase in enumerate(segment["visual_plan"]["phases"]):
                    phase_dir = run / task / f"phase_{index:02d}"
                    prompt = local.contact_review_prompt("tripod")
                    phase_dir.mkdir(parents=True, exist_ok=True)
                    (phase_dir / "prompt.txt").write_text(prompt)
                    samples = sorted({phase["start_sample"], (phase["start_sample"] + phase["end_sample"]) // 2, phase["end_sample"]})
                    images = []
                    for ordinal in samples:
                        image = directory / "localized_visual_inputs/segment_000" / f"sample_{ordinal:02d}.jpg"
                        images.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(image.read_bytes()).decode()}})
                    write_json(phase_dir / "samples.json", samples)
                    raw = pipeline.call_vlm([*images, {"type": "text", "text": prompt}], "Inspect object contact and return JSON only.")
                    (phase_dir / "response.txt").write_text(raw)
                    review = pipeline._extract_json_value(raw)
                    local.contact_parts_from_review(review)
                    phase["contact_review_before_refinement"] = phase.get("contact_review")
                    phase["contact_review"] = review
                    phase["contact_review_source"] = str(phase_dir / "response.txt")
                    local.include_contact_parts(phase)
                    print(json.dumps({"task": task, "phase": index, "contact_review": review}), flush=True)
                local.validate_phases(segment["visual_plan"], len(segment["sample_frames"]), False, require_contact_review=True)
                actions = local.compile_actions(segment["visual_plan"], np.asarray(segment["sample_frames"]), segment["segment"])
                all_actions.extend(actions)
            # Keep the already resolved temporal rules, not another refinement pass.
            for new, old in zip(all_actions, plan["actions"], strict=True):
                new["keyframe_function"] = old["keyframe_function"]
            plan["actions"] = all_actions
            plan["contact_refinement"] = {"method": "focused_contact_only_english", "audit": str(run / task)}
            signals, _, fps = pipeline.load_retargeting_bundle_signals(DATA / "omomo/bundles" / task / "input/omomo_gt_sequence.npz")
            resolved = local.execute_local_plan(plan, signals, fps)
            assert [e["windows"] for e in resolved["events"]] == [e["windows"] for e in result["events"]]
            result["events"] = resolved["events"]
            result["generation_metadata"]["contact_inclusion_policy"] = "motion_parts_union_identified_contact_parts_v1"
            result["generation_metadata"]["contact_refinement"] = plan["contact_refinement"]
            write_json(directory / "localized_phase_cache/contact_refined_segment_000.json", {
                "source": "focused_contact_only_english",
                "visual_plan": plan["segments"][0]["visual_plan"],
                "audit": str(run / task),
            })
            write_json(event_file, plan)
            write_json(file, result)
            checked = validate_output(root, "omomo", task)
            write_json(run / task / "validation.json", checked)
            if checked["status"] != "validated":
                raise ValueError(checked)


if __name__ == "__main__":
    main()
