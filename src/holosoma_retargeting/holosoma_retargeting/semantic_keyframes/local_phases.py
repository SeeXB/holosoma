"""Visual phase planning followed by bounded, reproducible GT refinement.

The VLM sees numbered samples within short clips. It labels intervals and
anatomy, not arbitrary numeric predicates. Explicit visual boundaries remain
visual estimates; physical refinements require evidence and never fall back
to a nearest threshold. Legacy dynamic plans retain their original executor.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS

from . import pipeline as legacy
from .body_constraints import apply_body_part_constraints

SCHEMA = "holosoma.localized_visual_phases.v1"
OBJECT_CONTACT_PROMPT_VERSION = "hand_wrist_forearm_review_en_v2"
REFINEMENTS = {"visual_boundary", "object_rise", "object_fall", "pelvis_rise", "pelvis_fall"}

PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]


def phase_prompt(sample_count, body_only, object_name=None, body_part_constraints=None):
    scene = (
        "These are chronological skeleton samples of a short human motion clip. Each numbered sample has "
        "X/Z and Y/Z projections. Red means anatomical left; blue means anatomical right. There is no object. "
        if body_only
        else f"These are chronological video samples of a person interacting with a {object_name}. "
    )
    prompt = scene + (
        f"The clip contains {sample_count} numbered samples. Identify visible action phases; return JSON only. "
        "Do not copy example actions or invent off-screen actions. Repeated walking/running may form one phase. "
        f"Return {'1 to 6' if body_only else '2 to 6'} phases with these fields: "
        '{"phases":[{"action":"english_snake_case_name","start_sample":1,"end_sample":4,'
        '"body_parts":["allowed_anatomy_name"],"criticality_level":3,"rationale":"visible evidence"}]}. '
        f"Sample indices must satisfy 1 <= start_sample < end_sample <= {sample_count}. "
        "Sort phases chronologically. Adjacent phases may share a boundary sample or start on the next sample. "
        "Do not overlap phases. Cover most visible motion; static leading/trailing samples may be omitted. "
        f"body_parts must use only these 18 exact strings: {sorted(legacy.MAPPABLE_BODY_PARTS)}. "
        "Do not use vague labels such as arm, leg, right_arm, left_arm, right_leg or left_leg. "
        "shoulder means shoulder; elbow includes the elbow and forearm; wrist means wrist; hand means palm/fingers; "
        "hip means hip; knee means knee; ankle means ankle; pelvis means pelvis; waist means waist/trunk. "
        "Select the parts that matter in the visible phase, rather than copying the entire vocabulary. "
        "Write rationale in English and explain visual evidence for the selected parts, not just the action name. "
        "criticality_level must be 1, 2, 3 or 4. Only describe visual phases and anatomy; do not generate thresholds, "
        "formulas or a refinement field. The program will resolve boundaries against local trajectory signals. "
    )
    if not body_only:
        prompt += (
            "Check object contact explicitly: body_parts must include the important parts grasping, supporting "
            "or carrying the object, not only the joints involved in walking or bending. "
            "HAND means the palm/finger region. G1 has a hand collision link even without actuated fingers; "
            "the lack of finger motion is not a reason to omit hand. Include hand when the palm/fingers grasp, "
            "support, push or pull the object. WRIST means the wrist region; wrist is not a substitute for hand. "
            "FOREARM means the surface between elbow and wrist; use elbow for this region in the current body_parts "
            "vocabulary. Elbow flexion alone does not prove forearm contact. Wrist movement alone does not prove "
            "wrist contact. These regions may contact simultaneously, but evaluate each from the images. "
            "Add a contact_review object to EVERY phase, with hand, wrist, forearm and evidence fields. "
            "Each of hand/wrist/forearm must be exactly left, right, both, none or uncertain, describing which "
            "anatomical side of that region contacts the object. Use the person's left/right, not image left/right. "
            "Two-handed grasping requires hand=both and both left_hand and right_hand in body_parts. "
            "A left/right/both contact judgment must include the corresponding hand/wrist/elbow labels in body_parts. "
            "Write evidence in English: describe how the object is supported and distinguish palm, wrist and forearm "
            "contact. If hand is not selected, explain why. Use uncertain for occlusion or insufficient detail, "
            "not none. Do not invent hand contact just because this is a carrying task. "
            "While the object is held in the hands, retain hand even during walking. Reassess after release. "
        )
        if object_name == "tripod":
            prompt += (
                "The tripod is the thin yellow/orange structure next to the blue person, not part of the body. "
                "Follow that object across the full-size frames as well as the overview. Describe how it moves "
                "relative to the person and inspect the hands near its shaft. Do not replace object interaction "
                "with a generic walking description. Distinguish contact from occlusion; do not guess a side. "
                "A statement about knee bending is not evidence about hand-object contact."
            )
    if not body_only and body_part_constraints is not None:
        if body_part_constraints.get("mode") != "bimanual_upper_limbs_v1":
            raise ValueError("Unsupported body-part constraint")
        prompt += (
            "Task metadata supplied by the user confirms a bimanual interaction. "
            "For each selected upper-limb region (hand, wrist, elbow/forearm, shoulder), "
            "include BOTH left and right body_parts, even when one side is occluded. "
            "Do not turn a known two-handed task into a one-handed task because of the camera view. "
            "Keep raw visual contact evidence distinct from this user-specified bilateral constraint. "
            "Do not add an unrelated anatomical region merely to make the labels symmetric. "
        )
    return prompt


def contact_parts_from_review(review):
    """Translate explicit contact judgments; never infer contact from the action name."""
    if not isinstance(review, dict):
        raise ValueError("each object phase requires contact_review for hand, wrist, forearm and evidence")
    parts = []
    for region, part in (("hand", "hand"), ("wrist", "wrist"), ("forearm", "elbow")):
        side = review.get(region)
        if side not in ("left", "right", "both", "none", "uncertain"):
            raise ValueError(f"contact_review.{region} must be left/right/both/none/uncertain")
        sides = ("left", "right") if side == "both" else (side,) if side in ("left", "right") else ()
        parts.extend(f"{s}_{part}" for s in sides)
    if not isinstance(review.get("evidence"), str) or len(review["evidence"].strip()) < 8:
        raise ValueError("contact_review.evidence must explain visible hand/wrist/forearm contact or uncertainty")
    return parts


def contact_review_prompt(object_name):
    """A focused second pass for contact evidence, without phase/timing generation."""
    return (
        f"Inspect these samples from one phase of a person interacting with a {object_name}. "
        "The person is blue and the object is yellow/orange. Focus on the object and the body surface "
        "that holds, pushes or supports it. Ignore walking joints. Distinguish the palm/fingers (hand), "
        "the wrist, and the forearm surface between elbow and wrist. Moving a joint does not prove contact. "
        "Return ONLY one JSON object with four keys: hand, wrist, forearm, evidence. "
        "For each of the first three keys, choose left, right, both, none or uncertain, using anatomical sides. "
        "Use uncertain for occlusion. evidence must be an English sentence explaining where the object "
        "touches or why you cannot determine contact. Do not generate phases, sample indices or body_parts."
    )


def include_contact_parts(phase):
    """Key parts are the union of motion-important parts and identified contact parts."""
    contacts = contact_parts_from_review(phase.get("contact_review"))
    original = phase.setdefault("body_parts_before_contact_review", list(phase["body_parts"]))
    phase["body_parts"] = list(dict.fromkeys([*original, *contacts]))
    phase["contact_body_parts"] = contacts
    phase["body_parts_added_from_contact_review"] = [p for p in contacts if p not in original]


def validate_contact_review(phase):
    missing = set(contact_parts_from_review(phase.get("contact_review"))) - set(phase["body_parts"])
    if missing:
        raise ValueError(f"contact_review identifies contact but body_parts omits {sorted(missing)}")


def validate_phases(value, count, body_only, *, require_contact_review=False):
    if isinstance(value, list):
        if len(value) == 1 and isinstance(value[0], dict) and "phases" in value[0]:
            value = value[0]
        else:
            value = {"phases": value}
    phases = value.get("phases") if isinstance(value, dict) else None
    if not isinstance(phases, list) or not (1 if body_only else 2) <= len(phases) <= min(12, count - 1):
        raise ValueError(
            "phases must contain 1-12 body-only phases or 2-12 object phases, with distinct sample boundaries"
        )
    previous = None
    for index, phase in enumerate(phases):
        if not isinstance(phase, dict):
            raise ValueError(f"phase {index} must be an object")
        name = phase.get("action")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
            raise ValueError(f"phase {index}: action must be snake_case")
        start, end = phase.get("start_sample"), phase.get("end_sample")
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start < end <= count
            or (previous is not None and not previous <= start <= previous + 1)
        ):
            raise ValueError(f"phase {index}: require start_sample={previous} and start < end <= {count}")
        parts = phase.get("body_parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(p, str) for p in parts):
            raise ValueError(f"phase {index}: body_parts must be a nonempty list of anatomy names")
        parts = list(dict.fromkeys(legacy.BODY_PART_NORMALIZATION.get(p, p) for p in parts))
        if set(parts) - legacy.MAPPABLE_BODY_PARTS:
            raise ValueError(f"phase {index}: unknown anatomy {set(parts) - legacy.MAPPABLE_BODY_PARTS}")
        phase["body_parts"] = parts
        if require_contact_review:
            include_contact_parts(phase)
            validate_contact_review(phase)
        refinement = phase.get("refinement") or "visual_boundary"
        phase["refinement"] = refinement
        if refinement not in REFINEMENTS or (body_only and refinement.startswith("object_")):
            raise ValueError(f"phase {index}: invalid refinement {refinement!r}")
        if type(phase.get("criticality_level")) is not int or phase["criticality_level"] not in (1, 2, 3, 4):
            raise ValueError(f"phase {index}: criticality_level must be 1-4")
        if not isinstance(phase.get("rationale"), str) or not phase["rationale"].strip():
            raise ValueError(f"phase {index}: provide visible evidence in rationale")
        previous = end
    if phases[-1]["end_sample"] - phases[0]["start_sample"] < 0.5 * (count - 1):
        raise ValueError("visual phases cover less than half the sampled clip; describe the remaining visible phases")
    if require_contact_review and not any(p["contact_body_parts"] for p in phases):
        raise ValueError(
            "No hand/wrist/forearm contact identified anywhere in this object-interaction clip. "
            "Reinspect the interaction object; if contact is not visible, retain uncertainty rather than "
            "inventing labels. An all-unresolved clip requires visual review, not automatic acceptance."
        )
    return value


def compile_actions(visual, sample_frames, segment_index):
    """Compile visual intervals into nonoverlapping local search brackets."""
    phases = visual["phases"]
    segment_tag = f"{segment_index:03d}" if isinstance(segment_index, int) else str(segment_index)
    anchors = [int(sample_frames[p["start_sample"] - 1]) for p in phases]
    actions = []
    for i, phase in enumerate(phases):
        sample = phase["start_sample"] - 1
        anchor = anchors[i]
        left = int(sample_frames[max(0, sample - 1)])
        right = int(sample_frames[min(len(sample_frames) - 1, sample + 1)])
        # Disjoint brackets prevent two adjacent boundaries collapsing to the
        # same frame. This restriction is explicit in the saved event plan.
        low = max(left, (anchors[i - 1] + anchor) // 2 + 1 if i else int(sample_frames[0]))
        high = min(right, (anchor + anchors[i + 1]) // 2 if i + 1 < len(phases) else int(sample_frames[-1]) - 1)
        rule = {
            "primitive": phase["refinement"],
            "visual_anchor_frame": anchor,
            "search_start_frame": low,
            "search_end_frame": high,
        }
        if rule["primitive"] != "visual_boundary":
            rule["minimum_vertical_speed_m_s"] = 0.02
        actions.append(
            {
                "action": f"segment_{segment_tag}_{i:02d}_{phase['action']}",
                "label": phase["action"],
                "body_parts": phase["body_parts"],
                "criticality_level": phase["criticality_level"],
                "rationale": phase["rationale"],
                "keyframe_function": {
                    "start": rule,
                    "end": {
                        "primitive": "next_phase_start_or_segment_end",
                        "segment_end_frame": int(sample_frames[phase["end_sample"] - 1]),
                    },
                },
                "visual_sample_interval": [phase["start_sample"], phase["end_sample"]],
            }
        )
    for action, phase in zip(actions, phases, strict=True):
        for field in (
            "contact_review", "contact_body_parts", "body_parts_before_contact_review",
            "body_parts_added_from_contact_review", "contact_review_source", "contact_review_before_refinement",
        ):
            if field in phase:
                action[field] = phase[field]
    return actions


def refine_actions_from_gt(actions, signals, fps):
    """Choose a physical refinement only with clear local directional evidence.

    Slow/static/horizontal phases keep their explicitly labeled visual anchor;
    this is a declared boundary method, not a failed predicate fallback.
    """
    signal = "object_height" if "object_height" in signals else "pelvis_height"
    prefix = "object" if "object_height" in signals else "pelvis"
    velocity = np.gradient(signals[signal]) * fps
    for action in actions:
        rule = action["keyframe_function"]["start"]
        if rule["primitive"] != "visual_boundary":
            continue
        lo, hi = rule["search_start_frame"], rule["search_end_frame"]
        hi = min(hi, action["keyframe_function"]["end"]["segment_end_frame"] - 1)
        rule["search_end_frame"] = hi
        local = velocity[lo : hi + 1]
        upward, downward = float(np.max(local)), float(np.max(-local))
        speed = max(upward, downward)
        # Require 4 cm/s and dominant direction, so small oscillations don't
        # turn a visually grounded boundary into a spurious height event.
        if speed >= 0.04 and min(upward, downward) <= 0.25 * speed:
            rule["primitive"] = prefix + ("_rise" if upward >= downward else "_fall")
            rule["minimum_vertical_speed_m_s"] = 0.04
        rule["selection_policy"] = "local_dominant_vertical_motion_else_explicit_visual_boundary"
    return actions


def resolve_start(rule, signals, fps):
    low, high = rule["search_start_frame"], rule["search_end_frame"]
    count = len(next(iter(signals.values())))
    if not 0 <= low <= high < count:
        raise ValueError(f"invalid local search bracket [{low},{high}] for {count} frames")
    primitive = rule["primitive"]
    if primitive == "visual_boundary":
        anchor = rule["visual_anchor_frame"]
        if not low <= anchor <= high:
            raise ValueError("visual anchor is outside its local bracket")
        return anchor
    if primitive not in REFINEMENTS:
        raise ValueError(f"unknown localized primitive {primitive}")
    signal = "object_height" if primitive.startswith("object_") else "pelvis_height"
    velocity = np.gradient(signals[signal]) * fps
    direction = 1 if primitive.endswith("rise") else -1
    signed = direction * velocity[low : high + 1]
    offset = int(np.argmax(signed))
    if signed[offset] < rule["minimum_vertical_speed_m_s"]:
        raise ValueError(
            f"{primitive} has no directional motion in local bracket [{low},{high}]; "
            f"peak requested signed velocity={float(signed[offset]):.4f} m/s. "
            "Reconsider visible phase boundary/refinement; no nearest-frame fallback is allowed."
        )
    return low + offset


def execute_local_plan(plan, signals, fps):
    if plan.get("schema") != SCHEMA or not plan.get("actions"):
        raise ValueError("invalid localized visual-phase plan")
    plan = apply_body_part_constraints(plan)
    starts = [resolve_start(a["keyframe_function"]["start"], signals, fps) for a in plan["actions"]]
    if any(b <= a for a, b in zip(starts, starts[1:])):
        raise ValueError(f"localized triggers must be strictly increasing: {starts}")
    events = []
    for i, action in enumerate(plan["actions"]):
        end = int(action["keyframe_function"]["end"]["segment_end_frame"])
        if i + 1 < len(starts):
            end = min(end, starts[i + 1])
        if end <= starts[i]:
            raise ValueError(f"{action['action']} has a zero-duration local phase")
        level = action["criticality_level"]
        events.append(
            {
                "event": action["action"],
                "action": action["action"],
                "label": action["label"],
                "body_parts": action["body_parts"],
                "confidence": 1.0,
                "criticality_level": level,
                "criticality": legacy.CRITICALITY_MAPPING[level],
                "criticality_rationale": action["rationale"],
                "rationale": action["rationale"],
                "failure_if_inaccurate": "The annotated local phase/body tracking may be misaligned.",
                "windows": [{"start_frame": starts[i], "trigger_frame": starts[i], "end_frame": end}],
                "keyframe_function": action["keyframe_function"],
                "boundary_source": "visual_sample"
                if action["keyframe_function"]["start"]["primitive"] == "visual_boundary"
                else "local_directional_velocity_peak",
            }
        )
    for event, action in zip(events, plan["actions"], strict=True):
        for field in (
            "contact_review", "contact_body_parts", "body_parts_before_contact_review",
            "body_parts_added_from_contact_review", "contact_review_source", "contact_review_before_refinement",
        ):
            if field in action:
                event[field] = action[field]
    result = {"fps": int(round(fps)), "semantic_mode": SCHEMA, "events": events}
    legacy.validate_semantic_keyframe_json(result)
    return result


def skeleton_images(joints, directory, task, count=64):
    """Chronological orthographic X/Z and Y/Z views, left red/right blue."""
    directory.mkdir(parents=True, exist_ok=True)
    centered = joints.copy()
    centered[:, :, :2] -= joints[:, :1, :2]
    scale = 70.0  # shared meters-to-pixel scale, never normalize each pose
    indices = np.linspace(0, len(joints) - 1, count).round().astype(int)
    content = []
    for sheet_index in range((count + 15) // 16):
        sheet = Image.new("RGB", (1200, 960), "white")
        draw = ImageDraw.Draw(sheet)
        for cell, index in enumerate(indices[sheet_index * 16 : (sheet_index + 1) * 16]):
            x0, y0 = (cell % 4) * 300, (cell // 4) * 240
            draw.text((x0 + 4, y0 + 4), f"{task} sample {sheet_index * 16 + cell + 1}/{count}", fill="black")
            for view, axis in enumerate((0, 1)):
                draw.text((x0 + view * 150 + 4, y0 + 22), ("X/Z", "Y/Z")[view], fill="black")
                draw.line((x0 + view * 150, y0 + 211, x0 + (view + 1) * 150, y0 + 211), fill="#bbbbbb")
                xy = np.stack(
                    [x0 + view * 150 + 75 + centered[index, :, axis] * scale, y0 + 211 - centered[index, :, 2] * scale],
                    axis=1,
                )
                for child, parent in enumerate(PARENTS):
                    if parent < 0:
                        continue
                    color = (
                        "#c92828"
                        if LAFAN_DEMO_JOINTS[child].startswith("Left")
                        else ("#245ac9" if LAFAN_DEMO_JOINTS[child].startswith("Right") else "#222222")
                    )
                    draw.line((*xy[parent], *xy[child]), fill=color, width=3)
                    x, y = xy[child]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        file = directory / f"skeleton_{sheet_index:02d}.jpg"
        sheet.save(file, quality=90)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(file.read_bytes()).decode()},
            }
        )
    return content


def video_images(video, frames, fps, directory):
    """Decode exact video samples at GT timestamps, label their image ordinals."""
    directory.mkdir(parents=True, exist_ok=True)
    content = []
    for ordinal, frame in enumerate(frames, 1):
        file = directory / f"sample_{ordinal:02d}.jpg"
        if not file.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    str(float(frame) / fps),
                    "-i",
                    str(video),
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=640:-2",
                    "-y",
                    str(file),
                ],
                check=True,
                capture_output=True,
            )
            with Image.open(file) as raw:
                labeled = Image.new("RGB", (raw.width, raw.height + 32), "white")
                labeled.paste(raw, (0, 32))
            ImageDraw.Draw(labeled).text((10, 10), f"Sample {ordinal} / {len(frames)}", fill="black")
            labeled.save(file, quality=90)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(file.read_bytes()).decode()},
            }
        )
    # Two contact sheets keep the complete short clip in view instead of
    # burying the final images behind twenty separate visual contexts.
    content = []
    for sheet_index in range((len(frames) + 9) // 10):
        sheet = Image.new("RGB", (1200, 640), "white")
        draw = ImageDraw.Draw(sheet)
        for cell, ordinal in enumerate(range(sheet_index * 10 + 1, min(len(frames), (sheet_index + 1) * 10) + 1)):
            with Image.open(directory / f"sample_{ordinal:02d}.jpg") as original:
                # Canonical renderer has white/gray background and colored body/object.
                pixels = np.asarray(original.convert("RGB"))[32:]
                mask = pixels.max(axis=2).astype(int) - pixels.min(axis=2).astype(int) > 45
                ys, xs = np.nonzero(mask)
                if len(xs):
                    crop = original.crop(
                        (
                            max(0, int(xs.min()) - 16),
                            max(32, int(ys.min()) + 16),
                            min(original.width, int(xs.max()) + 17),
                            min(original.height, int(ys.max()) + 49),
                        )
                    )
                else:
                    crop = original.crop((0, 32, original.width, original.height))
                crop.thumbnail((230, 285))
                x, y = (cell % 5) * 240, (cell // 5) * 320
                sheet.paste(crop, (x + (240 - crop.width) // 2, y + 30))
                draw.text((x + 8, y + 8), f"Sample {ordinal} / {len(frames)}", fill="black")
        path = directory / f"contact_sheet_{sheet_index:02d}.jpg"
        sheet.save(path, quality=92)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()},
            }
        )
    return content


def split_local_window(segment, start, stop):
    midpoint = (start + stop) // 2
    if midpoint - start < 2 or stop - midpoint < 2:
        raise ValueError("cannot split a local window into two meaningful clips")
    return [(segment + "a", start, midpoint), (segment + "b", midpoint, stop)]


def generate_localized_semantic_keyframes(
    *, video, bundle_file, output, audit_dir, max_repairs=2, body_part_constraints=None
):
    signals, _, fps = legacy.load_retargeting_bundle_signals(bundle_file)
    count = len(next(iter(signals.values())))
    body_only = "object_height" not in signals
    with np.load(bundle_file, allow_pickle=False) as bundle:
        joints = np.asarray(bundle["human_joints"]) if body_only else None
        object_name = str(bundle["object_name"].item()) if "object_name" in bundle.files else None
    # Short local clips retain repeated cycles without compressing the entire
    # several-minute LAFAN recording into a single 2-8-event plan.
    chunk_size = int(round(20 * fps)) if body_only else count
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    actions, segments = [], []
    jobs = []
    start = 0
    while start < count:
        stop = min(count, start + chunk_size)
        if count - stop == 1:
            stop = count
        jobs.append((f"{len(jobs):03d}", start, stop))
        start = stop
    cursor = 0
    while cursor < len(jobs):
        segment, start, stop = jobs[cursor]
        sample_count = 32 if body_only else 20
        if body_only and not segment.isdigit():
            sample_count = max(8, int(round((stop - start) / fps * 1.6)))
        frames = np.unique(np.linspace(start, stop - 1, sample_count).round().astype(int))
        images_dir = output.parent / "localized_visual_inputs" / f"segment_{segment}"
        if body_only:
            images = skeleton_images(joints[start:stop], images_dir, "body_motion", count=len(frames))
        else:
            images = video_images(video, frames, fps, images_dir)
            if object_name == "tripod":
                # Thin objects disappear in contact-sheet thumbnails. Reuse
                # existing full-size samples without adding new source data.
                for ordinal in (1, 4, 7, 10, 14, 18):
                    file = images_dir / f"sample_{ordinal:02d}.jpg"
                    if file.is_file():
                        images.append({"type": "image_url", "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(file.read_bytes()).decode()
                        }})
        base_prompt = phase_prompt(len(frames), body_only, object_name, body_part_constraints)
        (images_dir / "sample_frames.json").write_text(json.dumps(frames.tolist()))
        chunk_audit = audit_dir / f"segment_{segment}"
        chunk_audit.mkdir(parents=True, exist_ok=True)
        (chunk_audit / "prompt.txt").write_text(base_prompt)
        cached = output.parent / "localized_phase_cache" / f"segment_{segment}.json"
        signature = hashlib.sha256(
            (base_prompt + json.dumps(frames.tolist()) + str(fps)).encode()
            + np.stack([signals[name] for name in sorted(signals)]).tobytes()
        ).hexdigest()
        accepted = None
        if cached.is_file():
            stored = json.loads(cached.read_text())
            if stored.get("signature") == signature:
                accepted = validate_phases(
                    stored["visual_plan"], len(frames), body_only, require_contact_review=not body_only
                )
        split_cache = cached.with_suffix(".split.json")
        if accepted is None and split_cache.is_file():
            split = json.loads(split_cache.read_text())
            if split.get("signature") == signature:
                jobs[cursor : cursor + 1] = split_local_window(segment, start, stop)
                continue
        last_error = ""
        previous = None
        for attempt in range(max_repairs + 1):
            if accepted is not None:
                break
            prompt = base_prompt
            if previous is not None or last_error:
                prompt += (
                    f"\nYour previous attempt was rejected: {last_error}. Re-examine the images and return a "
                    "complete replacement JSON. Do not repeat the rejected structure. "
                )
                if not body_only:
                    prompt += (
                        "Every phase MUST contain contact_review with FOUR keys: hand, wrist, forearm, evidence. "
                        "The first three values are left/right/both/none/uncertain. The evidence value MUST be "
                        "a nonempty English sentence describing what touches the object or why it is uncertain. "
                        "Do not omit evidence. body_parts includes both motion-important and contacting parts."
                    )
            # Keep the contract after images as well: long image sequences
            # otherwise bury the instructions before the visual tokens.
            raw = legacy.call_vlm(
                [*images, {"type": "text", "text": prompt}], "Observe the images, follow the final English task instructions, and return JSON only."
            )
            (chunk_audit / f"attempt_{attempt}.txt").write_text(raw)
            try:
                previous = legacy._extract_json_value(raw)
                candidate = validate_phases(previous, len(frames), body_only, require_contact_review=not body_only)
                local_actions = refine_actions_from_gt(compile_actions(candidate, frames, segment), signals, fps)
                execute_local_plan({"schema": SCHEMA, "actions": local_actions}, signals, fps)
                accepted = candidate
            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)
        if accepted is None and body_only and stop - start > int(round(5 * fps)):
            cached.parent.mkdir(parents=True, exist_ok=True)
            split_note = {
                "signature": signature,
                "reason": last_error,
                "parent_frames": [start, stop - 1],
                "children": split_local_window(segment, start, stop),
            }
            split_cache.write_text(json.dumps(split_note, indent=2))
            (chunk_audit / "split_decision.json").write_text(json.dumps(split_note, indent=2))
            jobs[cursor : cursor + 1] = split_local_window(segment, start, stop)
            continue
        if accepted is None:
            raise RuntimeError(f"VLM failed to produce a valid localized plan for segment {segment}: {last_error}")
        compiled = refine_actions_from_gt(compile_actions(accepted, frames, segment), signals, fps)
        execute_local_plan({"schema": SCHEMA, "actions": compiled}, signals, fps)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps({"signature": signature, "visual_plan": accepted}, indent=2))
        actions.extend(compiled)
        segments.append(
            {
                "segment": segment,
                "start_frame": start,
                "end_frame": stop - 1,
                "sample_frames": frames.tolist(),
                "visual_plan": accepted,
            }
        )
        (audit_dir / "progress.json").write_text(
            json.dumps(
                {
                    "completed_segments": len(segments),
                    "total_segments": len(jobs),
                    "task": output.parent.name,
                }
            )
        )
        cursor += 1
    plan = {"schema": SCHEMA, "actions": actions, "segments": segments}
    if body_part_constraints is not None:
        plan["body_part_constraints"] = body_part_constraints
        plan = apply_body_part_constraints(plan)
    result = execute_local_plan(plan, signals, fps)
    issues = legacy.dynamic_resolution_validation_issues(result, count)
    if len(result["events"]) < 2 or issues:
        raise ValueError(f"localized complete-plan validation failed: {issues}")
    result["generation_metadata"] = {
        "model": os.environ["OPENAI_MODEL"],
        "temperature": 0,
        "seed": 0,
        "planning_mode": SCHEMA,
        "body_vocabulary_version": "g1_anatomy_v2",
        "source_bundle": str(bundle_file),
        "source_video": str(video),
        "segment_count": len(segments),
        "boundary_policy": "VLM numbered-image phases; explicit visual boundaries or strict local signed-velocity peaks",
        "visual_review": "pending",
    }
    if body_part_constraints is not None:
        result["generation_metadata"]["body_part_constraints"] = body_part_constraints
    if not body_only:
        result["generation_metadata"]["contact_prompt_version"] = OBJECT_CONTACT_PROMPT_VERSION
        result["generation_metadata"]["contact_inclusion_policy"] = "motion_parts_union_identified_contact_parts_v1"
    legacy._write_json(output.with_name(output.stem + ".event_plan.json"), plan)
    legacy._write_json(output, result)
    return result
