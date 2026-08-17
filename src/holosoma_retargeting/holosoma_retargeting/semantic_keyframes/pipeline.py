"""Generate semantic keyframes from video and pre-retargeting motion data.

The VLM is deliberately restricted to selecting semantic events and a small
set of declarative trigger rules. Frame indices are computed locally from the
human/object motion, making the result reproducible and auditable.
"""

from __future__ import annotations

import base64
import copy
import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

ROLE_ORDER = ("start", "approach", "contact", "lift", "carry_mid", "arrive", "place", "release")
CRITICALITY_MAPPING = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
MAPPABLE_BODY_PARTS = frozenset({"pelvis", "left_hand", "right_hand", "left_foot", "right_foot"})
SIGNALS = {
    "left_hand_box_surface_distance",
    "right_hand_box_surface_distance",
    "min_hand_box_surface_distance",
    "max_hand_box_surface_distance",
    "pelvis_box_distance",
    "object_height",
    "object_speed",
    "object_progress",
    "object_goal_distance",
    "hand_mean_speed",
    "pelvis_speed",
}
PRIMITIVES = {
    "global_minimum",
    "global_maximum",
    "local_minimum",
    "local_maximum",
    "threshold_crossing_down",
    "threshold_crossing_up",
    "sustained_below",
    "sustained_above",
}
THRESHOLDS = {
    "contact_enter",
    "contact_exit",
    "lift_enter",
    "lift_exit",
    "near_goal",
    "released",
    "moving",
    "still",
    "approach_distance",
    "approach_speed",
}

# This grammar is the executable contract between the VLM and local signal
# evaluator. It prevents a visually plausible plan from choosing a physically
# unrelated trigger, such as detecting lift from hand distance alone.
ROLE_TRIGGER_REQUIREMENTS = {
    "start": ("global_minimum", "object_progress", None),
    "approach": ("local_minimum", "pelvis_speed", None),
    "contact": ("threshold_crossing_down", "max_hand_box_surface_distance", "contact_enter"),
    "lift": ("threshold_crossing_up", "object_height", "lift_enter"),
    "carry_mid": ("threshold_crossing_up", "object_progress", "moving"),
    "arrive": ("threshold_crossing_down", "object_goal_distance", "near_goal"),
    "place": ("threshold_crossing_down", "object_height", "lift_exit"),
    "release": ("threshold_crossing_up", "max_hand_box_surface_distance", "released"),
}
ROLE_END_REQUIREMENTS = {
    "start": ("local_minimum", "pelvis_speed", None),
    "approach": ("threshold_crossing_down", "max_hand_box_surface_distance", "contact_enter"),
    "contact": ("threshold_crossing_up", "object_height", "lift_enter"),
    "lift": ("threshold_crossing_up", "object_progress", "moving"),
    "carry_mid": ("threshold_crossing_down", "object_goal_distance", "near_goal"),
    "arrive": ("threshold_crossing_down", "object_height", "lift_exit"),
    "place": ("threshold_crossing_up", "max_hand_box_surface_distance", "released"),
    "release": ("global_maximum", "max_hand_box_surface_distance", None),
}

def load_env_file(path: Path) -> None:
    """Load missing environment variables from a simple dotenv file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def sample_video_frames(video: Path, count: int) -> list[dict[str, Any]]:
    """Sample the source video exactly as the original GMR pipeline did."""
    if count < 2:
        raise ValueError("--sample-count must be at least 2")
    with tempfile.TemporaryDirectory(prefix="semantic_frames_") as tmp:
        pattern = str(Path(tmp) / "frame_%03d.jpg")
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-vf",
                f"fps={count}/6.53,scale=512:-2",
                "-q:v",
                "3",
                pattern,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed to sample {video}: {result.stderr.strip()}")
        frames = sorted(Path(tmp).glob("frame_*.jpg"))
        if len(frames) < 2:
            raise RuntimeError(f"ffmpeg produced too few frames from {video}")
        if len(frames) > count:
            keep = np.linspace(0, len(frames) - 1, count).round().astype(int)
            frames = [frames[index] for index in keep]
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii")
                },
            }
            for frame in frames
        ]


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from a plain or Markdown-fenced VLM response."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ValueError("response contains no JSON object")
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("top-level response must be a JSON object")
    return value


def plan_validation_issues(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return field-addressable validation issues for a declarative plan.

    A repair is safe only when the event list and role identity are already
    valid.  Once that boundary is established, every issue names exactly one
    top-level event field so the local pipeline can freeze every other field.
    """
    issues: list[dict[str, Any]] = []

    def add(event: str | None, field: str, message: str, *, patchable: bool = True) -> None:
        key = (event, field)
        if any((issue["event"], issue["field"]) == key for issue in issues):
            return
        issues.append(
            {"event": event, "field": field, "message": message, "patchable": patchable}
        )

    events = plan.get("events")
    if not isinstance(events, list) or len(events) != len(ROLE_ORDER):
        add(
            None,
            "events",
            f"plan.events must contain exactly these roles: {list(ROLE_ORDER)}",
            patchable=False,
        )
        return issues
    if not all(isinstance(event, dict) for event in events):
        add(None, "events", "every plan.events item must be an object", patchable=False)
        return issues
    roles = [event.get("event") for event in events if isinstance(event, dict)]
    if tuple(roles) != ROLE_ORDER:
        add(
            None,
            "events",
            f"events must be ordered exactly as {list(ROLE_ORDER)}, got {roles}",
            patchable=False,
        )
        return issues

    forbidden = ("frame", "duration", "window", "python", "lambda", "==", "exec", "eval")
    for event in events:
        event_name = event["event"]
        for field, value in event.items():
            raw = json.dumps({field: value}).lower()
            if any(token in raw for token in forbidden):
                add(event_name, field, f"event {event_name!r}.{field} contains forbidden logic")
        for field in ("trigger", "end"):
            rule = event.get(field)
            if not isinstance(rule, dict):
                add(event_name, field, f"{event_name}.{field} must be an object")
                continue
            if rule.get("primitive") not in PRIMITIVES:
                add(event_name, field, f"unsupported primitive in {event_name}.{field}: {rule.get('primitive')!r}")
            if rule.get("signal") not in SIGNALS:
                add(event_name, field, f"unsupported signal in {event_name}.{field}: {rule.get('signal')!r}")
            if rule.get("threshold") is not None and rule.get("threshold") not in THRESHOLDS:
                add(event_name, field, f"unsupported threshold in {event_name}.{field}: {rule.get('threshold')!r}")
            after = rule.get("after")
            if after is not None and after not in ROLE_ORDER:
                add(event_name, field, f"unknown temporal relation in {event_name}.{field}: {after!r}")

        expected = ROLE_TRIGGER_REQUIREMENTS[event_name]
        trigger = event.get("trigger")
        if isinstance(trigger, dict):
            actual = (trigger.get("primitive"), trigger.get("signal"), trigger.get("threshold"))
            expected_after = ROLE_ORDER[ROLE_ORDER.index(event_name) - 1] if event_name != "start" else None
            if actual != expected or trigger.get("after") != expected_after:
                add(
                    event_name,
                    "trigger",
                    f"{event_name} trigger must be {(*expected, expected_after)}, got {(*actual, trigger.get('after'))}",
                )
        expected_end = ROLE_END_REQUIREMENTS[event_name]
        end = event.get("end")
        if isinstance(end, dict):
            actual_end = (end.get("primitive"), end.get("signal"), end.get("threshold"))
            if actual_end != expected_end or end.get("after") != event_name:
                add(
                    event_name,
                    "end",
                    f"{event_name} end must be {(*expected_end, event_name)}, got {(*actual_end, end.get('after'))}",
                )
        body_parts = event.get("body_parts")
        if not isinstance(body_parts, list) or not body_parts or not all(isinstance(part, str) for part in body_parts):
            add(event_name, "body_parts", f"{event_name}.body_parts must be a non-empty list of strings")
        else:
            unmappable = sorted(set(body_parts).difference(MAPPABLE_BODY_PARTS))
            if unmappable:
                add(event_name, "body_parts", f"{event_name} contains unmappable body parts: {unmappable}")
            if event_name in {"contact", "release"} and not any("hand" in part for part in body_parts):
                add(event_name, "body_parts", f"{event_name} must name at least one hand body part")
        level = event.get("criticality_level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in CRITICALITY_MAPPING:
            add(
                event_name,
                "criticality_level",
                f"{event_name}.criticality_level must be one of {sorted(CRITICALITY_MAPPING)}, got {level!r}",
            )
        if "criticality" in event:
            add(
                event_name,
                "criticality",
                "VLM plans must choose criticality_level; criticality is derived by the local pipeline",
            )
        for field in ("criticality_rationale", "failure_if_inaccurate"):
            value = event.get(field)
            if not isinstance(value, str) or not value.strip():
                add(event_name, field, f"{event_name}.{field} must be a non-empty string")
    return issues


def validate_plan(plan: dict[str, Any]) -> None:
    """Validate the constrained declarative event-plan schema."""
    issues = plan_validation_issues(plan)
    if issues:
        raise ValueError(str(issues[0]["message"]))


def build_prompt() -> str:
    """Build the VLM prompt that defines the declarative plan contract."""
    schema_example = {
        "events": [
            {
                "event": "<semantic_role>",
                "body_parts": ["<body_alias>"],
                "trigger": {
                    "primitive": "<allowed_primitive>",
                    "signal": "<allowed_signal>",
                    "threshold": "<optional_threshold>",
                    "after": None,
                },
                "end": {
                    "primitive": "<allowed_primitive>",
                    "signal": "<allowed_signal>",
                    "threshold": "<optional_threshold>",
                    "after": "<optional_previous_role>",
                },
                "criticality_level": 4,
                "criticality_rationale": "<why accurate retargeting of this event matters to task success>",
                "failure_if_inaccurate": "<likely downstream task failure caused by inaccurate geometry>",
                "rationale": "<brief visual rationale>",
            }
        ]
    }
    trigger_contract = {
        role: {
            "trigger": {
                "primitive": spec[0],
                "signal": spec[1],
                "threshold": spec[2],
                "after": ROLE_ORDER[index - 1] if index else None,
            },
            "end": {
                "primitive": ROLE_END_REQUIREMENTS[role][0],
                "signal": ROLE_END_REQUIREMENTS[role][1],
                "threshold": ROLE_END_REQUIREMENTS[role][2],
                "after": role,
            },
        }
        for index, (role, spec) in enumerate(ROLE_TRIGGER_REQUIREMENTS.items())
    }
    return (
        "You are planning semantic triggers for a human carrying a box. Analyze the sampled video frames.\n"
        f"Return JSON only. Generate one declarative plan for these ordered semantic roles: {list(ROLE_ORDER)}.\n"
        "Do not give frame numbers, fixed window lengths, arbitrary code, equality checks, or prose outside JSON.\n"
        f"body_parts must be a non-empty subset of {sorted(MAPPABLE_BODY_PARTS)}. Object/box is not a body part.\n"
        f"Every trigger/end must use one primitive from {sorted(PRIMITIVES)}; "
        f"one signal from {sorted(SIGNALS)}; and, when needed, one threshold name from {sorted(THRESHOLDS)}.\n"
        "Every event, including the final release event, must contain both trigger and end as non-null objects. "
        "Every trigger/end object must contain primitive, signal, threshold, and after; use JSON null when threshold "
        "or after is not needed. Never omit these keys.\n"
        "For the end.after field, name the same event whose end is being described. For trigger.after, use only the "
        "immediately prior role (or null for start). End rules describe a signal-based end, not a fixed duration.\n"
        "For every event choose exactly one integer criticality_level from 1, 2, 3, 4. Do not output a criticality "
        "float; the local pipeline maps the level to a fixed value. Criticality measures how strongly accurate "
        "retargeting of this event affects successful completion of the demonstrated task. It is not model "
        "confidence, frame-detection confidence, or motion magnitude. Judge it from (A) state-transition "
        "importance, such as not-holding to holding or supported to lifted; (B) failure sensitivity if the local "
        "body-object geometry is inaccurate; and (C) interaction specificity of the required body-object or "
        "body-environment relation.\n"
        "Fixed rubric: Level 1 is Auxiliary (semantic but normally recoverable); Level 2 is Moderately important "
        "(quality loss with recovery space); Level 3 is Important (inaccuracy likely affects this or later "
        "interaction); Level 4 is Task-critical (key state transition or strict contact/support relation whose "
        "failure likely breaks the task). The local mapping is 1->0.25, 2->0.50, 3->0.75, 4->1.00.\n"
        "Keep body_parts, criticality, and trigger timing conceptually separate. Include a concise "
        "criticality_rationale and failure_if_inaccurate for every event.\n"
        "Role-specific trigger/end contract (this is an executable interface requirement, not a video-specific "
        "answer). Copy every primitive, signal, threshold, and after value exactly; do not substitute values. The "
        "final release.end is therefore a fully populated global_maximum rule, never a null rule:\n"
        f"{json.dumps(trigger_contract, indent=2)}\n"
        "Schema example (placeholders only; do not copy placeholder values as an answer):\n"
        f"{json.dumps(schema_example, indent=2)}"
    )


def build_repair_prompt(
    plan: dict[str, Any],
    issues: list[dict[str, Any]],
    prior_repair_error: str | None = None,
) -> str:
    """Request only patches for invalid fields of an otherwise fixed plan."""
    allowed = [
        {"event": issue["event"], "field": issue["field"], "error": issue["message"]}
        for issue in issues
    ]
    prompt = (
        "Repair only the invalid fields listed below in a semantic event plan after inspecting the video.\n"
        "All unlisted fields and every event without a listed invalid field are immutable and already valid. "
        "Do not restate or regenerate the plan. Return JSON only, with exactly one repair for every listed "
        "event/field pair and no other pairs. Use action=replace with a value, or action=remove only when the "
        "invalid field must not exist.\n"
        "Repair response schema (placeholders only):\n"
        '{"repairs":[{"event":"<event>","field":"<field>","action":"replace","value":"<new value>"}]}\n'
        f"Allowed invalid fields:\n{json.dumps(allowed, indent=2)}\n"
        f"Immutable-source plan:\n{json.dumps(plan, indent=2)}"
    )
    if prior_repair_error:
        prompt += (
            "\nThe prior repair response was rejected without changing the immutable-source plan: "
            f"{prior_repair_error}. Return a corrected patch response."
        )
    return prompt


def apply_plan_repairs(
    plan: dict[str, Any],
    repair_payload: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply an exact allowlisted repair while freezing all validated fields."""
    if any(not issue.get("patchable", False) for issue in issues):
        raise ValueError("plan structure is invalid and cannot be repaired without changing validated events")
    allowed = {(str(issue["event"]), str(issue["field"])) for issue in issues}
    repairs = repair_payload.get("repairs")
    if not isinstance(repairs, list):
        raise ValueError("repair response must contain a repairs list")

    provided: list[tuple[str, str]] = []
    for repair in repairs:
        if not isinstance(repair, dict):
            raise ValueError("every repair must be an object")
        event = repair.get("event")
        field = repair.get("field")
        if not isinstance(event, str) or not isinstance(field, str):
            raise ValueError("every repair must name string event and field values")
        provided.append((event, field))
    if len(provided) != len(set(provided)):
        raise ValueError("repair response contains duplicate event/field pairs")
    if set(provided) != allowed:
        unexpected = sorted(set(provided).difference(allowed))
        missing = sorted(allowed.difference(provided))
        raise ValueError(f"repair paths must exactly match the allowlist; unexpected={unexpected}, missing={missing}")

    repaired = copy.deepcopy(plan)
    events_by_name = {event["event"]: event for event in repaired["events"]}
    for repair in repairs:
        event = events_by_name[repair["event"]]
        field = repair["field"]
        action = repair.get("action")
        if action == "replace":
            if "value" not in repair:
                raise ValueError(f"replace repair for {repair['event']}.{field} must contain value")
            event[field] = copy.deepcopy(repair["value"])
        elif action == "remove":
            if "value" in repair:
                raise ValueError(f"remove repair for {repair['event']}.{field} must not contain value")
            event.pop(field, None)
        else:
            raise ValueError(f"unsupported repair action for {repair['event']}.{field}: {action!r}")

    # The allowlist itself is enforced above. This second check makes the
    # immutability guarantee explicit and guards future changes to this code.
    original_by_name = {event["event"]: event for event in plan["events"]}
    for event_name, repaired_event in events_by_name.items():
        original_event = original_by_name[event_name]
        protected_fields = set(original_event).union(repaired_event).difference(
            field for allowed_event, field in allowed if allowed_event == event_name
        )
        for field in protected_fields:
            if original_event.get(field) != repaired_event.get(field):
                raise ValueError(f"repair modified protected field {event_name}.{field}")
    return repaired


def call_vlm(content: list[dict[str, Any]], prompt: str | None = None) -> str:
    """Call an OpenAI-compatible vision chat-completions endpoint."""
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL")
    if not (base_url and api_key and model):
        raise RuntimeError("OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL must be set (for example in .env)")

    prompt = prompt or build_prompt()
    request_body = json.dumps(
        {
            "model": model,
            "temperature": 0,
            "seed": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, *content]}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - configured OpenAI-compatible HTTP endpoint.
        base_url.rstrip("/") + "/chat/completions",
        data=request_body,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"VLM HTTP {exc.code}: {detail}") from exc
    return str(payload["choices"][0]["message"]["content"])


def load_smplx_signals(path: Path, model_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Reconstruct the SMPL-X body and compute the original local signals."""
    try:
        import smplx  # noqa: PLC0415 - keeps schema-only imports lightweight.
        import torch  # noqa: PLC0415 - keeps schema-only imports lightweight.
    except ImportError as exc:
        raise RuntimeError("SMPL-X signal extraction requires the smplx and torch packages") from exc

    data = np.load(path, allow_pickle=True)
    frames = len(data["trans"])
    gender = str(data["gender"].item())
    model = smplx.create(
        str(model_dir),
        "smplx",
        gender=gender,
        use_pca=False,
        num_betas=data["betas"].shape[-1],
    )
    with torch.no_grad():
        output = model(
            betas=torch.as_tensor(data["betas"], dtype=torch.float32).reshape(1, -1).repeat(frames, 1),
            global_orient=torch.as_tensor(data["root_orient"], dtype=torch.float32),
            body_pose=torch.as_tensor(data["pose_body"], dtype=torch.float32),
            transl=torch.as_tensor(data["trans"], dtype=torch.float32),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
            jaw_pose=torch.zeros(frames, 3),
            leye_pose=torch.zeros(frames, 3),
            reye_pose=torch.zeros(frames, 3),
            expression=torch.zeros(frames, 10),
            return_verts=False,
        )
    joints = output.joints.cpu().numpy()
    # SMPL-X canonical joints: wrists 20/21; hand proxies 25/40 are more distal.
    left_hand, right_hand, pelvis = joints[:, 25], joints[:, 40], joints[:, 0]
    obj = np.asarray(data["obj_com_pos"], dtype=np.float32)
    dist_l = np.linalg.norm(left_hand - obj, axis=1)
    dist_r = np.linalg.norm(right_hand - obj, axis=1)
    speed = lambda track: np.linalg.norm(np.gradient(track, axis=0), axis=1) * 30.0  # noqa: E731
    obj_delta = obj - obj[0]
    path_dir = obj[-1] - obj[0]
    norm = max(float(np.linalg.norm(path_dir)), 1e-6)
    progress = np.clip(obj_delta @ (path_dir / norm) / norm, 0.0, 1.0)
    signals = {
        "left_hand_box_surface_distance": dist_l,
        "right_hand_box_surface_distance": dist_r,
        "min_hand_box_surface_distance": np.minimum(dist_l, dist_r),
        "max_hand_box_surface_distance": np.maximum(dist_l, dist_r),
        "pelvis_box_distance": np.linalg.norm(pelvis - obj, axis=1),
        "object_height": obj[:, 2] - obj[0, 2],
        "object_speed": speed(obj),
        "object_progress": progress,
        "object_goal_distance": np.linalg.norm(obj - obj[-1], axis=1),
        "hand_mean_speed": 0.5 * (speed(left_hand) + speed(right_hand)),
        "pelvis_speed": speed(pelvis),
    }
    thresholds = {
        "contact_enter": float(np.quantile(signals["max_hand_box_surface_distance"], 0.18)),
        "contact_exit": float(np.quantile(signals["min_hand_box_surface_distance"], 0.72)),
        "lift_enter": float(np.quantile(signals["object_height"], 0.72)),
        "lift_exit": float(np.quantile(signals["object_height"], 0.30)),
        "near_goal": float(np.quantile(signals["object_goal_distance"], 0.18)),
        "released": float(np.quantile(signals["min_hand_box_surface_distance"], 0.80)),
        "moving": float(np.quantile(signals["object_speed"], 0.65)),
        "still": float(np.quantile(signals["object_speed"], 0.25)),
        "approach_distance": float(np.quantile(signals["pelvis_box_distance"], 0.45)),
        "approach_speed": float(np.quantile(signals["pelvis_speed"], 0.55)),
    }
    return signals, thresholds


def _rule_frame(
    rule: dict[str, Any],
    signals: dict[str, np.ndarray],
    thresholds: dict[str, float],
    minimum: int,
) -> int:
    values = signals[rule["signal"]]
    primitive = rule["primitive"]
    threshold = thresholds.get(rule.get("threshold"))
    valid = np.arange(len(values)) >= minimum
    if not valid.any():
        return len(values) - 1
    if primitive == "global_minimum":
        return int(np.arange(len(values))[valid][np.argmin(values[valid])])
    if primitive == "global_maximum":
        return int(np.arange(len(values))[valid][np.argmax(values[valid])])
    if primitive in {"local_minimum", "local_maximum"}:
        direction = 1 if primitive == "local_minimum" else -1
        candidates = (
            np.flatnonzero(
                (direction * values[1:-1] <= direction * values[:-2])
                & (direction * values[1:-1] <= direction * values[2:])
            )
            + 1
        )
        candidates = candidates[candidates >= minimum]
        if len(candidates):
            return int(candidates[np.argmin(direction * values[candidates])])
        return int(np.arange(len(values))[valid][np.argmin(direction * values[valid])])
    if threshold is None:
        raise ValueError(f"{primitive} requires a threshold")
    if primitive in {"threshold_crossing_down", "sustained_below"}:
        predicate = values <= threshold
    elif primitive in {"threshold_crossing_up", "sustained_above"}:
        predicate = values >= threshold
    else:
        raise ValueError(f"unsupported primitive {primitive}")
    if primitive.startswith("threshold_crossing"):
        candidates = np.flatnonzero(predicate & ~np.r_[False, predicate[:-1]])
    else:
        candidates = np.flatnonzero(predicate & np.r_[predicate[1:], False])
    candidates = candidates[candidates >= minimum]
    if len(candidates):
        return int(candidates[0])
    return int(np.arange(len(values))[valid][np.argmin(np.abs(values[valid] - threshold))])


def execute_plan(
    plan: dict[str, Any],
    signals: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Resolve a validated semantic plan into deterministic frame windows."""
    validate_plan(plan)
    events: list[dict[str, Any]] = []
    last_trigger = 0
    for event in plan["events"]:
        trigger = _rule_frame(event["trigger"], signals, thresholds, last_trigger)
        end = max(trigger, _rule_frame(event["end"], signals, thresholds, trigger))
        events.append(
            {
                "event": event["event"],
                "body_parts": event.get("body_parts", []),
                "confidence": 1.0,
                "criticality_level": event["criticality_level"],
                "criticality": CRITICALITY_MAPPING[event["criticality_level"]],
                "criticality_rationale": event["criticality_rationale"],
                "failure_if_inaccurate": event["failure_if_inaccurate"],
                "windows": [{"start_frame": trigger, "end_frame": end, "trigger_frame": trigger}],
                "trigger": event["trigger"],
                "end": event["end"],
                "rationale": event.get("rationale", ""),
            }
        )
        last_trigger = trigger
    return {"fps": 30, "events": events, "trigger_thresholds": thresholds}


def validate_semantic_keyframe_json(payload: dict[str, Any]) -> None:
    """Validate resolved windows, discrete criticality, and body aliases."""
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("resolved semantic JSON must contain a non-empty events list")
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("resolved semantic events must be objects")
        name = event.get("event")
        level = event.get("criticality_level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in CRITICALITY_MAPPING:
            raise ValueError(f"{name}.criticality_level must be one of {sorted(CRITICALITY_MAPPING)}")
        expected = CRITICALITY_MAPPING[level]
        criticality = event.get("criticality")
        if isinstance(criticality, bool) or not isinstance(criticality, (int, float)):
            raise ValueError(f"{name}.criticality must be numeric")
        if float(criticality) != expected:
            raise ValueError(f"{name}.criticality must equal {expected:.2f} for level {level}")
        body_parts = event.get("body_parts")
        if not isinstance(body_parts, list) or not body_parts:
            raise ValueError(f"{name}.body_parts must be non-empty")
        unmappable = sorted(set(body_parts).difference(MAPPABLE_BODY_PARTS))
        if unmappable:
            raise ValueError(f"{name} contains unmappable body parts: {unmappable}")
        windows = event.get("windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError(f"{name}.windows must be non-empty")
        for window in windows:
            start = window.get("start_frame")
            trigger = window.get("trigger_frame")
            end = window.get("end_frame")
            if any(isinstance(value, bool) or not isinstance(value, int) for value in (start, trigger, end)):
                raise ValueError(f"{name} window frames must be integers")
            if not start <= trigger <= end:
                raise ValueError(f"{name} requires start_frame <= trigger_frame <= end_frame")


def semantic_json_diff(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    """Build an auditable event-wise v1/v2 schema diff without altering either input."""
    fields = ("window", "trigger_frame", "body_parts", "criticality_level", "criticality")

    def event_view(event: dict[str, Any]) -> dict[str, Any]:
        windows = event.get("windows", [])
        first = windows[0] if windows else event
        values = {
            "window": [first.get("start_frame"), first.get("end_frame")],
            "trigger_frame": first.get("trigger_frame"),
            "body_parts": event.get("body_parts"),
            "criticality_level": event.get("criticality_level"),
            "criticality": event.get("criticality"),
        }
        return {field: values[field] for field in fields}

    old = {event["event"]: event_view(event) for event in v1.get("events", [])}
    new = {event["event"]: event_view(event) for event in v2.get("events", [])}
    rows = []
    for name in dict.fromkeys([*old, *new]):
        before = old.get(name)
        after = new.get(name)
        changed = [field for field in fields if (before or {}).get(field) != (after or {}).get(field)]
        rows.append({"event": name, "v1": before, "v2": after, "changed_fields": changed})
    return {"events": rows, "any_change": any(row["changed_fields"] for row in rows)}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _remove_stale_attempts(output: Path, last_attempt: int) -> None:
    """Keep the raw-attempt audit trail scoped to the completed generation run."""
    pattern = re.compile(rf"^{re.escape(output.stem)}\.vlm_attempt_(\d+)\.txt$")
    for path in output.parent.glob(f"{output.stem}.vlm_attempt_*.txt"):
        match = pattern.match(path.name)
        if match and int(match.group(1)) > last_attempt:
            path.unlink()


def generate_semantic_keyframes(
    *,
    video: Path,
    smplx_file: Path,
    output: Path,
    model_dir: Path = Path("assets/body_models"),
    sample_count: int = 12,
    max_repairs: int = 3,
    baseline: Path | None = None,
    diff_output: Path | None = None,
) -> dict[str, Any]:
    """Run the original VLM-to-SMPL-X semantic-keyframe pipeline."""
    load_env_file(Path(".env"))
    output.parent.mkdir(parents=True, exist_ok=True)
    images = sample_video_frames(video, sample_count)
    plan: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = []
    prior_repair_error: str | None = None
    repair_history: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(max_repairs + 1):
        prompt = build_prompt() if plan is None else build_repair_prompt(plan, issues, prior_repair_error)
        raw = call_vlm(images, prompt)
        raw_path = output.parent / f"{output.stem}.vlm_attempt_{attempt}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        try:
            if plan is None:
                candidate = extract_json(raw)
            else:
                candidate = apply_plan_repairs(plan, extract_json(raw), issues)
                repair_history.append(
                    {
                        "attempt": attempt,
                        "fields": [f"{issue['event']}.{issue['field']}" for issue in issues],
                    }
                )
            candidate_issues = plan_validation_issues(candidate)
            if candidate_issues:
                unpatchable = [issue for issue in candidate_issues if not issue.get("patchable", False)]
                if unpatchable:
                    raise RuntimeError(
                        "VLM plan has an invalid event structure that cannot be repaired without changing "
                        f"validated events: {unpatchable[0]['message']}"
                    )
                plan = candidate
                issues = candidate_issues
                prior_repair_error = None
                last_error = ValueError("; ".join(str(issue["message"]) for issue in issues))
                continue

            plan = candidate
            validate_plan(plan)
            plan_path = output.parent / f"{output.stem}.event_plan.json"
            _write_json(plan_path, plan)
            signals, thresholds = load_smplx_signals(smplx_file, model_dir)
            result = execute_plan(plan, signals, thresholds)
            result["generation_metadata"] = {
                "model": os.environ["OPENAI_MODEL"],
                "temperature": 0,
                "seed": 0,
                "sample_count": sample_count,
                "source_video": str(video),
                "source_smplx": str(smplx_file),
                "repair_policy": "field_only_immutable_valid_fields",
                "vlm_attempt_count": attempt + 1,
                "accepted_repairs": repair_history,
            }
            validate_semantic_keyframe_json(result)
            _write_json(output, result)
            if baseline is not None:
                old = json.loads(baseline.read_text(encoding="utf-8"))
                diff = semantic_json_diff(old, result)
                target = diff_output or output.with_name("semantic_v1_v2_diff.json")
                _write_json(target, diff)
            _remove_stale_attempts(output, attempt)
            return result
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            # A malformed/unauthorized patch never replaces the last locally
            # validated plan state. The next attempt receives the same exact
            # allowlist plus this formatting error.
            prior_repair_error = str(exc)
            last_error = exc
    raise RuntimeError(
        f"VLM failed to produce a valid trigger plan after field-only repairs; last error: {last_error}"
    )
