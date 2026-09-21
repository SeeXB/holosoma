"""Generate semantic keyframes from video and pre-retargeting motion data.

The VLM is deliberately restricted to selecting semantic events and a small
set of declarative trigger rules. Frame indices are computed locally from the
human/object motion, making the result reproducible and auditable.
"""

from __future__ import annotations

import base64
import copy
import hashlib
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
MAPPABLE_BODY_PARTS = frozenset(
    {"pelvis", "waist"}
    | {f"{side}_{part}" for side in ("left", "right")
       for part in ("hip", "knee", "ankle", "shoulder", "elbow", "wrist", "hand", "foot")}
)
BODY_PART_NORMALIZATION = {
    "torso": "waist", "left_forearm": "left_elbow", "right_forearm": "right_elbow",
}
# Names in canonical SMPL-H bundles and native LAFAN skeletons, respectively.
BODY_JOINT_NAMES = {
    "pelvis": ("Pelvis", "Hips"), "waist": ("Torso", "Spine"),
    **{f"{side}_{part}": (f"{short}_{smpl}", f"{long}{lafan}")
       for side, short, long in (("left", "L", "Left"), ("right", "R", "Right"))
       for part, smpl, lafan in (("hip", "Hip", "UpLeg"), ("knee", "Knee", "Leg"),
           ("ankle", "Ankle", "Foot"), ("foot", "Toe", "ToeBase"),
           ("shoulder", "Shoulder", "Arm"), ("elbow", "Elbow", "ForeArm"),
           ("wrist", "Wrist", "Hand"), ("hand", "Middle3", "Hand"))},
}
BODY_SIGNALS = {f"{part}_{metric}" for part in MAPPABLE_BODY_PARTS
                for metric in ("height", "speed")} | {"body_travel_progress"}

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
# Dataset-independent vocabulary exposed to the VLM.  Unlike the historical
# ``SIGNALS`` set above, these names do not encode a box-carry task.
DYNAMIC_SIGNALS = {
    "left_hand_object_distance",
    "right_hand_object_distance",
    "min_hand_object_distance",
    "max_hand_object_distance",
    "pelvis_object_distance",
    "left_hand_speed",
    "right_hand_speed",
    "hand_mean_speed",
    "pelvis_speed",
    "object_height",
    "object_vertical_speed",
    "object_speed",
    "object_displacement",
    "object_rotate",
    "object_progress",
    "object_goal_distance",
    "object_angular_speed",
}
DYNAMIC_SIGNALS.update(BODY_SIGNALS)
DYNAMIC_SIGNALS.update(f"{part}_object_distance" for part in MAPPABLE_BODY_PARTS)
DYNAMIC_PRIMITIVES = frozenset({
    "global_minimum",
    "global_maximum",
    "local_minimum",
    "local_maximum",
    "threshold_crossing_down",
    "threshold_crossing_up",
    "sustained_below",
    "sustained_above",
})
DYNAMIC_BODY_PARTS = MAPPABLE_BODY_PARTS
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
        # The repository keeps the retargeting credentials beside this
        # package, while the CLI is commonly launched from the workspace root.
        package_env = Path(__file__).resolve().parents[2] / ".env"
        if package_env == path or not package_env.exists():
            return
        path = package_env
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


def _extract_json_value(text: str) -> Any:
    """Extract the first complete JSON value from a VLM response."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    candidate = fenced.group(1).lstrip() if fenced else text
    # Dynamic plans may legally be returned as a top-level array.  Starting
    # extraction at the first ``{`` used to silently discard every action but
    # the first one in that case, turning a good VLM answer into an invalid
    # singleton object.
    starts = [index for index in (candidate.find("{"), candidate.find("[")) if index >= 0]
    if not starts:
        raise ValueError("response contains no JSON object or array")
    value, _ = json.JSONDecoder().raw_decode(candidate[min(starts):])
    return value


def extract_json(text: str) -> dict[str, Any]:
    """Extract one JSON object from a plain or Markdown-fenced VLM response."""
    value = _extract_json_value(text)
    if not isinstance(value, dict):
        raise ValueError("top-level response must be a JSON object")
    return value


DEFAULT_BODY_PARTS = {
    "start": ["pelvis"],
    "approach": ["pelvis"],
    "contact": ["left_hand", "right_hand"],
    "lift": ["left_hand", "right_hand"],
    "carry_mid": ["pelvis"],
    "arrive": ["pelvis"],
    "place": ["left_hand", "right_hand"],
    "release": ["left_hand", "right_hand"],
}


def normalize_plan_body_parts(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove VLM-only anatomy aliases before executable validation.

    The runtime deliberately supports a small body-part vocabulary.  Keeping
    valid aliases and role-specific defaults makes the boundary safe while
    leaving all trigger/end/criticality decisions untouched.
    """
    normalized = copy.deepcopy(plan)
    for event in normalized.get("events", []):
        if not isinstance(event, dict):
            continue
        role = event.get("event")
        parts = event.get("body_parts")
        if isinstance(parts, list):
            valid = [part for part in parts if part in MAPPABLE_BODY_PARTS]
            if role in {"contact", "release"} and not any("hand" in part for part in valid):
                valid = list(DEFAULT_BODY_PARTS.get(role, []))
            event["body_parts"] = valid or list(DEFAULT_BODY_PARTS.get(role, []))
    return normalized


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


class VLMQuotaError(RuntimeError):
    """Provider quota rejection; this does not establish the account balance."""

    def __init__(self, message, provider_error=None):
        super().__init__(message)
        self.provider_error = provider_error or {}


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
        if exc.code == 429 and "insufficient_quota" in detail:
            try:
                error = json.loads(detail).get("error", {})
            except (ValueError, AttributeError):
                error = {}
            if not isinstance(error, dict):
                error = {}
            diagnostic = {"http_status": exc.code}
            for field in ("code", "type", "message", "param"):
                if isinstance(error.get(field), str):
                    value = error[field].replace(api_key, "[REDACTED]")
                    diagnostic[field] = value[:500] if "data:image" not in value else "[omitted echoed image data]"
            raise VLMQuotaError("VLM HTTP 429: insufficient_quota; account balance not determined", diagnostic) from exc
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


def load_retargeting_bundle_signals(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """Compute semantic signals from an explicit joint/object trajectory bundle.

    This is the dataset-independent counterpart to :func:`load_smplx_signals`.
    It consumes named joints plus an aligned metric object trajectory and never
    opens an OMOMO/InterMimic tensor.  The signal definitions and quantile
    thresholds intentionally match the existing semantic-plan resolver.
    """
    with np.load(path, allow_pickle=False) as data:
        if "object_poses_wxyz_xyz" not in data.files:
            return load_body_bundle_signals(data)
        required = {
            "human_joints",
            "object_poses_wxyz_xyz",
            "smplh_joint_names",
            "frame_ids",
            "fps",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"retargeting bundle is missing keys: {missing}")
        joints = np.asarray(data["human_joints"], dtype=np.float64)
        object_poses = np.asarray(data["object_poses_wxyz_xyz"], dtype=np.float64)
        names = [str(value) for value in np.asarray(data["smplh_joint_names"]).tolist()]
        frame_ids = np.asarray(data["frame_ids"])
        fps = float(np.asarray(data["fps"]).item())
        # OMOMO renderer bundles contain ``sequence_name``.  Require the
        # explicit post-permutation marker so native BodyModel.Jtr columns can
        # never again be interpreted using retarget joint names.
        is_omomo_bundle = "sequence_name" in data.files
        human_joint_layout = (
            str(np.asarray(data["human_joint_layout"]).item())
            if "human_joint_layout" in data.files
            else None
        )
    if is_omomo_bundle and human_joint_layout != "smplh_retarget_v1":
        raise ValueError(
            "OMOMO bundle lacks verified smplh_retarget_v1 joint ordering; "
            "regenerate it with tools/omomo_cari4d_renderer/prepare_sequence.py"
        )
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"human_joints must have shape [T,J,3], got {joints.shape}")
    if object_poses.shape != (len(joints), 7) or frame_ids.shape != (len(joints),):
        raise ValueError("human joints, object poses, and frame IDs must have identical frame counts")
    if len(names) != joints.shape[1] or len(set(names)) != len(names):
        raise ValueError("smplh_joint_names must uniquely name every joint column")
    if not np.isfinite(joints).all() or not np.isfinite(object_poses).all():
        raise ValueError("retargeting bundle contains NaN/Inf")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"invalid bundle fps: {fps}")
    for name in ("L_Middle3", "R_Middle3", "Pelvis"):
        if name not in names:
            raise ValueError(f"retargeting bundle lacks required joint {name!r}")

    left_hand = joints[:, names.index("L_Middle3")]
    right_hand = joints[:, names.index("R_Middle3")]
    pelvis = joints[:, names.index("Pelvis")]
    obj = object_poses[:, 4:7]
    dist_l = np.linalg.norm(left_hand - obj, axis=1)
    dist_r = np.linalg.norm(right_hand - obj, axis=1)

    def speed(track: np.ndarray) -> np.ndarray:
        return np.linalg.norm(np.gradient(track, axis=0), axis=1) * fps

    def angular_speed(quaternion_wxyz: np.ndarray) -> np.ndarray:
        # Quaternion sign is arbitrary; use the shortest angular increment.
        q = quaternion_wxyz / np.maximum(
            np.linalg.norm(quaternion_wxyz, axis=1, keepdims=True), 1.0e-8
        )
        dots = np.sum(q[1:] * q[:-1], axis=1)
        angles = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
        return np.r_[angles[0] if len(angles) else 0.0, angles] * fps

    def cumulative_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
        """Unsigned 3-D orientation travel from the first frame, in radians."""
        q = quaternion_wxyz / np.maximum(
            np.linalg.norm(quaternion_wxyz, axis=1, keepdims=True), 1.0e-8
        )
        dots = np.sum(q[1:] * q[:-1], axis=1)
        increments = 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))
        return np.r_[0.0, np.cumsum(increments)]

    obj_delta = obj - obj[0]
    path_dir = obj[-1] - obj[0]
    norm = max(float(np.linalg.norm(path_dir)), 1e-6)
    progress = np.clip(obj_delta @ (path_dir / norm) / norm, 0.0, 1.0)
    object_displacement = np.linalg.norm(obj - obj[0], axis=1)
    object_vertical_speed = np.abs(np.gradient(obj[:, 2])) * fps
    signals = {
        # Historical names say "surface" although the registered resolver uses
        # the canonical object-frame origin.  Keep names/semantics stable here.
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
        "left_hand_object_distance": dist_l,
        "right_hand_object_distance": dist_r,
        "min_hand_object_distance": np.minimum(dist_l, dist_r),
        "max_hand_object_distance": np.maximum(dist_l, dist_r),
        "pelvis_object_distance": np.linalg.norm(pelvis - obj, axis=1),
        "left_hand_speed": speed(left_hand),
        "right_hand_speed": speed(right_hand),
        "object_vertical_speed": object_vertical_speed,
        "object_displacement": object_displacement,
        "object_rotate": cumulative_rotation(object_poses[:, :4]),
        "object_angular_speed": angular_speed(object_poses[:, :4]),
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
    body_signals = compute_body_signals(joints, names, fps, obj)
    # Preserve the legacy hand/pelvis definitions used by existing plans.
    signals.update({key: value for key, value in body_signals.items() if key not in signals})
    return signals, thresholds, fps


def compute_body_signals(joints, names, fps, obj=None):
    """Metric world-space body signals; cumulative travel is not a frame counter."""
    signals = {}
    for part, aliases in BODY_JOINT_NAMES.items():
        name = next((name for name in aliases if name in names), None)
        if name is None:
            continue
        track = joints[:, names.index(name)]
        signals[f"{part}_height"] = track[:, 2]
        signals[f"{part}_speed"] = np.linalg.norm(np.gradient(track, axis=0), axis=1) * fps
        if obj is not None:
            signals[f"{part}_object_distance"] = np.linalg.norm(track - obj, axis=1)
    travel = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(joints, axis=0), axis=2).mean(axis=1))]
    signals["body_travel_progress"] = travel / max(float(travel[-1]), 1e-8)
    return signals


def load_body_bundle_signals(data):
    """Load a explicitly marked Z-up, full-rate body-only bundle (e.g. LAFAN)."""
    if str(np.asarray(data["human_joint_layout"]).item()) != "lafan_z_up_v1":
        raise ValueError("body-only bundle requires lafan_z_up_v1 joint layout")
    joints = np.asarray(data["human_joints"], dtype=float)
    names = [str(name) for name in data["smplh_joint_names"]]
    fps = float(np.asarray(data["fps"]).item())
    if (joints.ndim != 3 or joints.shape[1:] != (len(names), 3) or len(joints) < 2
            or len(set(names)) != len(names) or not np.isfinite(joints).all()
            or not np.isfinite(fps) or fps <= 0
            or np.asarray(data["frame_ids"]).shape != (len(joints),)):
        raise ValueError("invalid body-only trajectory shape, names, values, frame IDs or fps")
    signals = compute_body_signals(joints, names, fps)
    if "pelvis_height" not in signals:
        raise ValueError("body-only bundle requires a named pelvis joint")
    return signals, {}, fps


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


def dynamic_plan_validation_issues(plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the task-specific VLM action/function contract.

    An action is intentionally not selected from a global role list.  The VLM
    may name the observed action (for example ``drag_suitcase`` or
    ``grasp_handle``), while its frame judgment remains a small declarative
    function over locally computed motion signals.
    """
    issues: list[dict[str, Any]] = []

    def add(action: str | None, field: str, message: str) -> None:
        issues.append({"action": action, "field": field, "message": message})

    actions = plan.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= 12:
        add(None, "actions", "actions must contain between 1 and 12 task-specific actions")
        return issues
    names: list[str] = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            add(None, f"actions[{index}]", "every action must be an object")
            continue
        name = action.get("action")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", name):
            add(str(name) if name is not None else None, "action", "action must be a unique snake_case name")
            continue
        if name in names:
            add(name, "action", "action names must be unique")
        names.append(name)
        parts = action.get("body_parts")
        if not isinstance(parts, list) or not parts or not all(isinstance(part, str) for part in parts):
            add(name, "body_parts", "body_parts must be a non-empty list of strings")
        elif set(parts).difference(DYNAMIC_BODY_PARTS):
            add(name, "body_parts", f"unmappable body parts: {sorted(set(parts).difference(DYNAMIC_BODY_PARTS))}")
        function = action.get("keyframe_function")
        if not isinstance(function, dict):
            add(name, "keyframe_function", "keyframe_function must contain start and end rules")
            continue
        for endpoint in ("start", "end"):
            rule = function.get(endpoint)
            if not isinstance(rule, dict):
                add(name, f"keyframe_function.{endpoint}", "rule must be an object")
                continue
            primitive = rule.get("primitive")
            signal = rule.get("signal")
            if primitive not in DYNAMIC_PRIMITIVES:
                add(name, f"keyframe_function.{endpoint}", f"unsupported primitive: {primitive!r}")
            if signal not in DYNAMIC_SIGNALS:
                add(name, f"keyframe_function.{endpoint}", f"unsupported signal: {signal!r}")
            threshold = rule.get("threshold")
            needs_threshold = primitive in {"threshold_crossing_down", "threshold_crossing_up", "sustained_below", "sustained_above"}
            if needs_threshold:
                if not isinstance(threshold, dict) or threshold.get("kind") != "quantile":
                    add(name, f"keyframe_function.{endpoint}", "threshold rules require {kind: quantile, q: number}")
                elif isinstance(threshold.get("q"), bool) or not isinstance(threshold.get("q"), (int, float)) or not 0.0 <= float(threshold["q"]) <= 1.0:
                    add(name, f"keyframe_function.{endpoint}", "quantile q must be in [0, 1]")
            elif threshold is not None:
                add(name, f"keyframe_function.{endpoint}", "non-threshold primitives must use threshold=null")
        if isinstance(function.get("start"), dict) and function.get("start") == function.get("end"):
            add(
                name,
                "keyframe_function",
                "start and end rules are identical and would collapse the action; use distinct onset/exit predicates",
            )
        level = action.get("criticality_level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in CRITICALITY_MAPPING:
            add(name, "criticality_level", "criticality_level must be one of 1, 2, 3, 4")
        for field in ("rationale", "criticality_rationale", "failure_if_inaccurate"):
            if not isinstance(action.get(field), str) or not action[field].strip():
                add(name, field, f"{field} must be a non-empty string")
    return issues


def validate_dynamic_plan(plan: dict[str, Any]) -> None:
    issues = dynamic_plan_validation_issues(plan)
    if issues:
        raise ValueError(str(issues[0]["message"]))


def build_dynamic_prompt(
    object_name: str | None = None,
    available_signals=None,
    body_only=False,
) -> str:
    object_hint = f" The manipulated object category is {object_name!r}." if object_name else ""
    json_example = (
        {
            "actions": [
                {
                    "action": "example_lift_object",
                    "body_parts": ["left_hand", "right_hand"],
                    "keyframe_function": {
                        "start": {
                            "primitive": "threshold_crossing_up",
                            "signal": "object_height",
                            "threshold": {"kind": "quantile", "q": 0.3},
                        },
                        "end": {
                            "primitive": "threshold_crossing_up",
                            "signal": "object_height",
                            "threshold": {"kind": "quantile", "q": 0.8},
                        },
                    },
                    "criticality_level": 4,
                    "rationale": "The object visibly rises while supported by both hands.",
                    "criticality_rationale": "Accurate support is essential during the lift.",
                    "failure_if_inaccurate": "The object may not be lifted safely.",
                },
                {
                    "action": "example_rotate_object",
                    "body_parts": ["left_hand", "right_hand"],
                    "keyframe_function": {
                        "start": {
                            "primitive": "threshold_crossing_up",
                            "signal": "object_rotate",
                            "threshold": {"kind": "quantile", "q": 0.2},
                        },
                        "end": {
                            "primitive": "threshold_crossing_up",
                            "signal": "object_rotate",
                            "threshold": {"kind": "quantile", "q": 0.8},
                        },
                    },
                    "criticality_level": 3,
                    "rationale": "The supported object visibly changes orientation.",
                    "criticality_rationale": "The target orientation affects later placement.",
                    "failure_if_inaccurate": "The object may be placed in the wrong orientation.",
                },
                {
                    "action": "example_place_object",
                    "body_parts": ["left_hand", "right_hand"],
                    "keyframe_function": {
                        "start": {
                            "primitive": "threshold_crossing_down",
                            "signal": "object_height",
                            "threshold": {"kind": "quantile", "q": 0.8},
                        },
                        "end": {
                            "primitive": "threshold_crossing_down",
                            "signal": "object_height",
                            "threshold": {"kind": "quantile", "q": 0.2},
                        },
                    },
                    "criticality_level": 4,
                    "rationale": "The supported object visibly descends to its resting surface.",
                    "criticality_rationale": "Accurate placement is essential for a stable final state.",
                    "failure_if_inaccurate": "The object may be released before it is stably placed.",
                },
            ]
        }
        if not body_only else
        {
            "actions": [
                {
                    "action": "example_body_motion",
                    "body_parts": ["pelvis"],
                    "keyframe_function": {
                        "start": {
                            "primitive": "threshold_crossing_up",
                            "signal": "pelvis_speed",
                            "threshold": {"kind": "quantile", "q": 0.2},
                        },
                        "end": {
                            "primitive": "threshold_crossing_down",
                            "signal": "pelvis_speed",
                            "threshold": {"kind": "quantile", "q": 0.2},
                        },
                    },
                    "criticality_level": 2,
                    "rationale": "The pelvis begins and then finishes a visible motion.",
                    "criticality_rationale": "The interval captures the central body transition.",
                    "failure_if_inaccurate": "The motion phase may be retargeted at the wrong time.",
                }
            ]
        }
    )
    anatomy = (
        f" Allowed body_parts are exactly {sorted(DYNAMIC_BODY_PARTS)}. "
        "Choose anatomically precise parts visible in the images, not a fixed list. "
        "Hand means palm/fingers; wrist means the wrist joint; elbow includes the forearm link. "
        "Forearm support/clamping should name left_elbow/right_elbow, not automatically hand. "
        "Waist means the trunk/waist. Include only parts important to the observed action. "
        "Explain the visual evidence for those body parts in rationale. Do not copy an example body part list. "
    )
    scene = (
        "Analyze chronological front/side skeleton views of a body-only motion clip. "
        "There is no manipulated object. Red is left, blue is right; views are pelvis-centered horizontally. "
        "Describe locomotion, jumps, falls, getting up, dance, fighting or other visible body actions. "
        "For long repetitive clips summarize 2 to 8 successive phases, not every cycle. "
        "body_travel_progress is normalized cumulative mean joint travel (not elapsed time); "
        "it can delimit early/middle/late movement phases. Heights are world Z and speeds are world m/s. "
        if body_only else
        "Analyze the entire rerendered human-object video and produce a task-specific semantic action plan."
    )
    interaction_hint = (
        "Use only body-motion signals; do not infer a manipulated object. " if body_only else
        f"{object_hint} Do not assume a box-carry task: distinguish lifting, carrying, dragging, pushing, "
        "placing, rotating, supporting, or any other actions actually visible. "
    )
    return anatomy + scene + interaction_hint + (
        "HARD EXECUTABLE CONTRACT: every threshold-crossing or sustained rule requires a quantile threshold. "
        "Prefer threshold_crossing_down, threshold_crossing_up, sustained_below, or sustained_above for "
        "multi-action transitions; use global/local extrema only for a genuinely unique visible peak or valley. "
        "Example threshold rule: "
        '{"primitive":"threshold_crossing_down","signal":"pelvis_speed",'
        '"threshold":{"kind":"quantile","q":0.25}}. '
        "Return one JSON object only. Do not return Python, pseudocode, Markdown or prose outside JSON. "
        "Here is a complete event-plan JSON example. It demonstrates the required shape and how distinct "
        "physical actions use distinct signals. It is not the answer: replace every example action, body part, "
        "rule and explanation with actions actually visible in this video:\n"
        + json.dumps(json_example, indent=2) + "\n"
        "Choose 2 to 8 distinct actions in the temporal order observed; action names must be unique snake_case. "
        "Each action must describe a meaningful state or interaction transition and name the body parts that matter. "
        "The keyframe_function is a declarative interval predicate: start finds the action's first keyframe and "
        "end finds its last keyframe on a separate GT trajectory. Use no frame numbers, durations, Python, lambda, "
        "code, or arbitrary thresholds. A threshold must be a quantile object with q in [0,1]. For global/local "
        "min/max rules use threshold=null. The allowed primitives are "
        f"{sorted(DYNAMIC_PRIMITIVES)}; allowed signals are {sorted(available_signals if available_signals is not None else DYNAMIC_SIGNALS)}. "
        "Use object_height only when the video shows vertical lifting; for dragging/sliding, use object_progress, "
        "object_displacement, object_speed, and hand-object distance instead. For a visible orientation change, "
        "use object_rotate to represent cumulative rotation progress; object_angular_speed only represents how fast "
        "the object is rotating at an instant. Do not describe rotation using contact distance alone. For lowering "
        "or placement after a lift, object_height must cross downward: use threshold_crossing_down at a higher "
        "quantile for onset and at a lower quantile for completion. Do not use upward height crossings for placing. "
        "Call an action dragging or sliding only when the video shows the object remaining supported by the ground; "
        "visible off-ground transport is lifting or carrying even when horizontal displacement is large. "
        "Every action must include all fields "
        "shown above. Temporal correctness is essential: choose the earliest visible onset after the previous action "
        "and the last frame of that action, not a later repetition or terminal pose. For transition actions, prefer "
        "threshold_crossing_up/down or sustained_above/below with a quantile over a global minimum/maximum; global "
        "extrema are appropriate only for an unambiguous single peak/valley. Do not use a global minimum of a "
        "distance/progress signal to mean 'first contact', because the same minimum can occur again at release. "
        "Do not emit duplicate zero-length actions unless the video clearly shows an instantaneous event. The resolved "
        "intervals must remain in the listed action order and cover the visible interaction rather than only the final "
        "few frames. Never use identical start and end rules for one action. "
        "When repeated extrema or crossings cannot separate successive phases, body_travel_progress "
        "is monotonic cumulative mean joint travel and permits distinct increasing quantile crossings; "
        "choose quantiles to reflect the visible phases, rather than repeatedly selecting the same peak. "
    )


def normalize_dynamic_plan(plan: Any) -> dict[str, Any]:
    """Normalize harmless VLM naming variants without inventing frame labels."""
    if isinstance(plan, list):
        plan = {"actions": plan}
    if not isinstance(plan, dict):
        return {"actions": plan}
    normalized = copy.deepcopy(plan)
    actions = normalized.get("actions")
    if actions is None and isinstance(normalized.get("events"), list):
        actions = normalized["events"]
        normalized["actions"] = actions
    if not isinstance(actions, list):
        return normalized
    for action in actions:
        if not isinstance(action, dict):
            continue
        if "action" not in action and isinstance(action.get("event"), str):
            action["action"] = action.pop("event")
        if "keyframe_function" not in action:
            function = action.get("frame_judgment") or action.get("function")
            if isinstance(function, dict):
                action["keyframe_function"] = function
        function = action.get("keyframe_function")
        if isinstance(function, dict):
            if "start" not in function and "enter" in function:
                function["start"] = function.pop("enter")
            if "end" not in function and "exit" in function:
                function["end"] = function.pop("exit")
            # Accept the legacy top-level rule names only as an input alias.
            if "start" not in function and isinstance(action.get("trigger"), dict):
                function["start"] = action["trigger"]
            if "end" not in function and isinstance(action.get("end"), dict):
                function["end"] = action["end"]
            for endpoint in ("start", "end"):
                rule = function.get(endpoint)
                if isinstance(rule, dict):
                    primitive = rule.get("primitive")
                    if primitive in {"global_minimum", "global_maximum", "local_minimum", "local_maximum"}:
                        rule["threshold"] = None
        parts = action.get("body_parts")
        if isinstance(parts, list):
            expanded: list[str] = []
            for part in parts:
                if isinstance(part, str):
                    expanded.extend(token.strip() for token in part.split("|") if token.strip())
            action["body_parts"] = list(dict.fromkeys(BODY_PART_NORMALIZATION.get(part, part) for part in expanded))
    return normalized


def _dynamic_rule_frame(rule: dict[str, Any], values: np.ndarray, minimum: int) -> int:
    primitive = rule["primitive"]
    valid = np.arange(len(values)) >= minimum
    if not valid.any():
        return len(values) - 1
    if primitive in {"global_minimum", "global_maximum", "local_minimum", "local_maximum"}:
        return _rule_frame({"primitive": primitive, "signal": "_dynamic"}, {"_dynamic": values}, {}, minimum)
    threshold_spec = rule.get("threshold")
    threshold = float(np.quantile(values, float(threshold_spec["q"])))
    if primitive in {"threshold_crossing_down", "sustained_below"}:
        predicate = values <= threshold
    else:
        predicate = values >= threshold
    if primitive.startswith("threshold_crossing"):
        # A crossing needs two observed samples: the previous sample must be on
        # the other side of the threshold.  Treating an already-true predicate
        # at frame 0 as a crossing fabricates a keyframe before any motion.
        candidates = np.flatnonzero(predicate[1:] & ~predicate[:-1]) + 1
    else:
        candidates = np.flatnonzero(predicate & np.r_[predicate[1:], False])
    candidates = candidates[candidates >= minimum]
    if len(candidates):
        return int(candidates[0])
    return int(np.arange(len(values))[valid][np.argmin(np.abs(values[valid] - threshold))])


def execute_dynamic_plan(
    plan: dict[str, Any],
    signals: dict[str, np.ndarray],
    fps: float = 30.0,
) -> dict[str, Any]:
    """Evaluate each VLM-supplied interval function on aligned GT signals."""
    if plan.get("schema") == "holosoma.trajectory_event_program.v1":
        from .trajectory_events import execute_event_program
        return execute_event_program(plan, signals, fps)
    if plan.get("schema") == "holosoma.localized_visual_phases.v1":
        from .local_phases import execute_local_plan
        return execute_local_plan(plan, signals, fps)
    validate_dynamic_plan(plan)
    # Resolve action onsets in their declared temporal order.  An action's
    # end predicate must not become the lower bound for the next onset: noisy
    # GT signals can make an otherwise sensible end crossing recur near the
    # clip tail and collapse every later action to that frame.  Endpoints are
    # resolved independently and then clipped at the following onset, which
    # preserves both VLM functions while producing an ordered phase partition.
    starts: list[int] = []
    previous_start = 0
    for action in plan["actions"]:
        start_rule = action["keyframe_function"]["start"]
        start = _dynamic_rule_frame(
            start_rule,
            signals[start_rule["signal"]],
            previous_start,
        )
        starts.append(start)
        previous_start = start

    events: list[dict[str, Any]] = []
    for action_index, action in enumerate(plan["actions"]):
        function = action["keyframe_function"]
        start_rule = function["start"]
        end_rule = function["end"]
        start = starts[action_index]
        raw_end = max(start, _dynamic_rule_frame(end_rule, signals[end_rule["signal"]], start))
        end = (
            min(raw_end, starts[action_index + 1])
            if action_index + 1 < len(starts)
            else raw_end
        )
        events.append({
            "event": action["action"],
            "action": action["action"],
            "body_parts": action["body_parts"],
            "confidence": 1.0,
            "criticality_level": action["criticality_level"],
            "criticality": CRITICALITY_MAPPING[action["criticality_level"]],
            "criticality_rationale": action["criticality_rationale"],
            "failure_if_inaccurate": action["failure_if_inaccurate"],
            "windows": [{"start_frame": start, "end_frame": end, "trigger_frame": start}],
            "trigger": start_rule,
            "end": end_rule,
            "keyframe_function": function,
            "rationale": action["rationale"],
        })
    return {"fps": int(round(fps)), "events": events, "semantic_mode": "dynamic_vlm_functions"}


def dynamic_resolution_validation_issues(
    result: dict[str, Any],
    frame_count: int,
) -> list[dict[str, Any]]:
    """Reject declarative plans that become meaningless after GT evaluation.

    Schema validation alone cannot detect several valid-looking VLM outputs,
    such as four different actions whose global extrema all resolve to the
    final frame.  Keep this check task-agnostic: it inspects only the resolved
    temporal support and sends actionable feedback to the next VLM repair
    attempt without inventing any action labels or frame indices.
    """
    issues: list[dict[str, Any]] = []
    events = result.get("events", [])
    if not events:
        return [{"action": None, "field": "resolved_events", "message": "the resolved plan has no events"}]

    resolved: list[tuple[str, int, int]] = []
    for event in events:
        windows = event.get("windows", [])
        if not windows:
            continue
        window = windows[0]
        resolved.append((str(event.get("event")), int(window["start_frame"]), int(window["end_frame"])))
    if not resolved:
        return [{"action": None, "field": "resolved_events", "message": "the resolved plan has no windows"}]

    duplicates: dict[tuple[int, int], list[str]] = {}
    for name, start, end in resolved:
        duplicates.setdefault((start, end), []).append(name)
    for (start, end), names in duplicates.items():
        if len(names) > 1:
            issues.append({
                "action": None,
                "field": "resolved_windows",
                "message": (
                    f"distinct actions {names} all resolve to duplicate window [{start}, {end}]; "
                    "choose different transition predicates instead of a shared global extremum"
                ),
            })

    duplicate_starts: dict[int, list[str]] = {}
    for name, start, _end in resolved:
        duplicate_starts.setdefault(start, []).append(name)
    for start, names in duplicate_starts.items():
        if len(names) > 1:
            issues.append({
                "action": None,
                "field": "resolved_windows",
                "message": (
                    f"distinct actions {names} all start at frame {start}; choose different onset "
                    "predicates so semantic transitions have positive temporal support"
                ),
            })

    zero_duration = [name for name, start, end in resolved if end <= start]
    if len(zero_duration) > max(1, len(resolved) // 2):
        issues.append({
            "action": None,
            "field": "resolved_windows",
            "message": (
                f"too many actions resolve to zero duration ({zero_duration}); use threshold crossings or "
                "sustained predicates so meaningful phases have temporal support"
            ),
        })

    last_frame = max(0, frame_count - 1)
    first_start = resolved[0][1]
    final_end = resolved[-1][2]
    minimum_span = max(2, int(np.ceil(0.10 * last_frame)))
    if len(resolved) >= 2 and final_end - first_start < minimum_span:
        issues.append({
            "action": None,
            "field": "resolved_windows",
            "message": (
                f"the full plan resolves only to frames [{first_start}, {final_end}] in a {frame_count}-frame "
                "clip; select predicates that cover the visible interaction rather than only one short region"
            ),
        })
    if len(resolved) >= 2 and first_start > int(0.80 * last_frame):
        issues.append({
            "action": resolved[0][0],
            "field": "keyframe_function.start",
            "message": (
                f"the first action begins at frame {first_start} of {frame_count}; avoid a late global extremum "
                "and select the earliest visible interaction onset"
            ),
        })
    return issues


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
    smplx_file: Path | None = None,
    bundle_file: Path | None = None,
    output: Path,
    model_dir: Path = Path("assets/body_models"),
    sample_count: int = 12,
    max_repairs: int = 3,
    dynamic: bool = True,
    baseline: Path | None = None,
    diff_output: Path | None = None,
) -> dict[str, Any]:
    """Run VLM planning and deterministic local resolution.

    ``smplx_file`` preserves the original SMPL-X route.  ``bundle_file`` is a
    dataset-independent route for OMOMO/CARI4D archives and is preferred for
    rerendered OMOMO sequences because it uses the exact aligned GT joints and
    object trajectory without reconstructing SMPL-X a second time.
    """
    if (smplx_file is None) == (bundle_file is None):
        raise ValueError("provide exactly one of smplx_file or bundle_file")
    if bundle_file is not None and dynamic:
        return generate_dynamic_semantic_keyframes(
            video=video,
            bundle_file=bundle_file,
            output=output,
            sample_count=sample_count,
            max_repairs=max_repairs,
        )
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
                candidate = normalize_plan_body_parts(extract_json(raw))
            else:
                repair_payload = extract_json(raw)
                # Some compatible VLMs occasionally echo a second already
                # valid field alongside the requested patch.  Ignore such
                # out-of-allowlist echoes while retaining the strict
                # field-only application and immutable-plan checks below.
                allowed_repairs = {
                    (str(issue["event"]), str(issue["field"])) for issue in issues
                }
                repairs = repair_payload.get("repairs")
                if isinstance(repairs, list):
                    repair_payload["repairs"] = [
                        repair for repair in repairs
                        if isinstance(repair, dict)
                        and (str(repair.get("event")), str(repair.get("field")))
                        in allowed_repairs
                    ]
                candidate = apply_plan_repairs(plan, repair_payload, issues)
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
            if bundle_file is not None:
                signals, thresholds, bundle_fps = load_retargeting_bundle_signals(bundle_file)
            else:
                signals, thresholds = load_smplx_signals(smplx_file, model_dir)
                bundle_fps = 30.0
            result = execute_plan(plan, signals, thresholds)
            result["fps"] = int(round(bundle_fps))
            result["generation_metadata"] = {
                "model": os.environ["OPENAI_MODEL"],
                "temperature": 0,
                "seed": 0,
                "sample_count": sample_count,
                "source_video": str(video),
                "source_smplx": None if smplx_file is None else str(smplx_file),
                "source_bundle": None if bundle_file is None else str(bundle_file),
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


def generate_dynamic_semantic_keyframes(
    *,
    video: Path,
    bundle_file: Path,
    output: Path,
    sample_count: int = 12,
    max_repairs: int = 3,
    audit_dir: Path | None = None,
    image_content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate task-specific VLM actions and resolve their functions on GT.

    The VLM never sees or emits GT frame indices.  It emits only action names,
    body aliases, and a constrained start/end predicate.  Quantile thresholds
    are evaluated independently on the supplied trajectory, so two tasks can
    choose different actions and obtain different keyframe intervals.
    """
    load_env_file(Path(".env"))
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_dir = audit_dir or output.parent
    audit_dir.mkdir(parents=True, exist_ok=True)
    images = image_content if image_content is not None else sample_video_frames(video, sample_count)
    with np.load(bundle_file, allow_pickle=False) as bundle:
        object_name = str(np.asarray(bundle["object_name"]).item()) if "object_name" in bundle.files else None
    signals, _thresholds, fps = load_retargeting_bundle_signals(bundle_file)
    frame_count = len(next(iter(signals.values())))
    base_prompt = build_dynamic_prompt(object_name, set(signals) & DYNAMIC_SIGNALS,
                                       body_only="object_height" not in signals)
    plan: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    previous_candidate: dict[str, Any] | None = None
    issues: list[dict[str, Any]] = []
    last_error: Exception | None = None
    for attempt in range(max_repairs + 1):
        prompt = base_prompt
        if attempt:
            prompt += (
                "\nYour previous response failed local validation. Here is the exact normalized response: "
                f"{json.dumps(previous_candidate, ensure_ascii=False)}\n"
                "Regenerate the complete plan and correct every listed field. A threshold primitive with "
                "threshold:null is invalid: replace null with {\"kind\":\"quantile\",\"q\":<a number in [0,1]>} "
                "chosen for the action visible in the video. Fix these issues: "
                f"{json.dumps(issues, ensure_ascii=False)}"
            )
        raw = call_vlm(images, prompt)
        raw_path = audit_dir / f"{output.stem}.vlm_attempt_{attempt}.txt"
        raw_path.write_text(raw, encoding="utf-8")
        try:
            candidate = normalize_dynamic_plan(_extract_json_value(raw))
            previous_candidate = candidate
            issues = dynamic_plan_validation_issues(candidate)
            if issues:
                last_error = ValueError("; ".join(str(issue["message"]) for issue in issues))
                continue
            validate_dynamic_plan(candidate)
            candidate_result = execute_dynamic_plan(candidate, signals, fps=fps)
            issues = dynamic_resolution_validation_issues(candidate_result, frame_count)
            if issues:
                last_error = ValueError("; ".join(str(issue["message"]) for issue in issues))
                continue
            plan = candidate
            result = candidate_result
            break
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            issues = [{"action": None, "field": "plan", "message": str(exc)}]
    if plan is None or result is None:
        raise RuntimeError(
            f"VLM failed to produce a valid dynamic action plan after {max_repairs + 1} attempts; "
            f"last error: {last_error}"
        )
    plan_path = output.parent / f"{output.stem}.event_plan.json"
    _write_json(plan_path, plan)
    result["generation_metadata"] = {
        "model": os.environ["OPENAI_MODEL"],
        "temperature": 0,
        "seed": 0,
        "sample_count": sample_count,
        "source_video": str(video),
        "source_bundle": str(bundle_file),
        "planning_mode": "dynamic_vlm_actions_and_functions",
        "body_vocabulary_version": "g1_anatomy_v2",
        "base_prompt_sha256": hashlib.sha256(base_prompt.encode()).hexdigest(),
        "visual_source_kind": "skeleton_views" if "object_height" not in signals else "video",
        "threshold_policy": "per-signal trajectory quantiles",
        "vlm_attempt_count": attempt + 1,
    }
    validate_semantic_keyframe_json(result)
    _write_json(output, result)
    _remove_stale_attempts(audit_dir / output.name, attempt)
    return result
