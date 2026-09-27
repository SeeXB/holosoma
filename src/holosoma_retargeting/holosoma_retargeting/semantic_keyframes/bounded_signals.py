"""Full-video semantic intervals with bounded, auditable GT change detection.

The VLM supplies approximate times, anatomy and a signal/transition choice.
It never supplies physical thresholds or quantiles. Unresolved boundaries are
retained as diagnostics, never substituted with a nearest value/visual frame.
Defaults are a versioned experimental detector, not calibrated contact labels.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import pipeline as legacy

SCHEMA = "holosoma.bounded_signal_plan.v1"
PROMPT_VERSION = "visual_intervals_gt_changes_v1"
TRANSITIONS = frozenset({
    "increase_onset", "increase_offset", "decrease_onset", "decrease_offset",
    "motion_onset", "motion_offset", "clip_start", "clip_end",
})


@dataclass(frozen=True)
class DetectorConfig:
    """One fixed configuration for all clips; floors describe resolvable motion."""

    smoothing_seconds: float = 0.10
    dwell_seconds: float = 0.10
    baseline_seconds: float = 0.20
    context_seconds: float = 0.75
    noise_multiplier: float = 4.0
    exit_ratio: float = 0.5
    linear_speed_floor: float = 0.02
    angular_speed_floor: float = 0.02
    normalized_rate_floor: float = 0.005

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) or v <= 0 for v in values.values()):
            raise ValueError("detector parameters must be finite and positive")
        if not 0 < self.exit_ratio < 1:
            raise ValueError("exit_ratio must lie between 0 and 1")


def _is_speed(name: str) -> bool:
    return name.endswith("_speed")


def signal_guide(available: set[str]) -> str:
    special = {
        "object_height": "object-origin world height relative to its first frame; not ground clearance",
        "object_displacement": "distance of object origin from its first-frame position, not total path length",
        "object_rotate": "accumulated unsigned 3D rotation; never decreases, not orientation or height",
        "object_progress": "clipped projection towards final position; can decrease; not video time",
        "object_goal_distance": "distance of object origin from final-frame position; not contact",
        "body_travel_progress": "normalized accumulated whole-body joint travel; never decreases; not time",
        "object_vertical_speed": "absolute vertical speed; lifting AND lowering are positive",
        "object_angular_speed": "unsigned rotation speed",
        "hand_mean_speed": "average world speed of the two finger/hand joints",
    }
    lines = []
    for name in sorted(available):
        if name in special:
            description = special[name]
        elif name.endswith("_object_distance"):
            description = "distance of named joint to OBJECT ORIGIN, NOT surface distance or contact force"
        elif name.endswith("_height"):
            description = "world Z height of named joint; rise/fall describes that joint only"
        elif _is_speed(name):
            description = "unsigned 3D world speed of named joint/object; does not show motion direction"
        else:
            raise ValueError(f"undocumented signal: {name}")
        operations = "motion_onset, motion_offset" if _is_speed(name) else (
            "increase_onset, increase_offset" if name in {"object_rotate", "body_travel_progress"}
            else "increase_onset, increase_offset, decrease_onset, decrease_offset"
        )
        lines.append(f"- {name}: {description}. Allowed: {operations}.")
    return "\n".join(lines)


def build_prompt(available: set[str], duration: float) -> str:
    max_width = min(2.0, max(0.8, 0.4 * duration))
    return f"""Watch the entire supplied video (time 0 to {duration:.6f} seconds).
Identify its actual visible interaction phases in temporal order. Do not infer a
predefined action sequence from the object category. Do not invent actions after
the clip ends. Repetitions may have separate phases; simultaneous actions may
overlap. Include visible approach/release only when relevant, not as mandatory phases.

For EACH phase, identify involved body regions and TWO approximate boundary time
ranges in seconds. These are visual localization estimates, NOT GT frame indices.
Choose a GT signal and transition that can test EACH boundary within its range.
Do NOT estimate physical values, thresholds, quantiles, fractions of signal range,
formulas, extrema or code. The program derives thresholds from local GT noise and
detects sustained changes with hysteresis. It returns unresolved if unsupported.

Return ONLY this JSON structure; strings in angle brackets describe fields:
{{"schema":"{SCHEMA}","actions":[{{
 "action":"<short descriptive action name>",
 "body_parts":["<allowed body region>"],
 "criticality_level":3,
 "rationale":"<what is visibly happening and why chosen signals fit>",
 "start":{{"time_range_s":["<earliest plausible second>","<latest plausible second>"],
            "signal":"<available signal>","transition":"<allowed transition>"}},
 "end":{{"time_range_s":["<earliest plausible second>","<latest plausible second>"],
          "signal":"<available signal>","transition":"<allowed transition>"}}
}}]}}
All time range entries must be JSON NUMBERS, not strings. Each range is within
[0,{duration:.6f}], with low <= high and width <= {max_width:.6f} seconds.
Use a genuine visual uncertainty interval (often several tenths of a second),
not a single guessed frame or an entire clip. Start midpoint must precede end
midpoint. Give start midpoints in chronological order. There is no target number
of phases and no requirement to force a partition or cover irrelevant inactivity.

TRANSITIONS:
- increase_onset: a position/distance/progress signal starts sustained INCREASE;
  increase_offset: that sustained INCREASE stops. These operate on its signed derivative.
- decrease_onset / decrease_offset: sustained DECREASE starts/stops (signed derivative).
- motion_onset / motion_offset: an unsigned SPEED signal changes from quiet to
  moving / moving to quiet. These test the speed itself, not acceleration.
- clip_start / clip_end: ONLY when a phase is already underway at the beginning
  or still underway at the end. signal=null and range=[0,0] / [{duration:.6f},{duration:.6f}].
  These are explicitly video-censored boundaries, NOT GT-detected transitions.
  clip_start is allowed only for the first phase start; clip_end only for the final phase end.
Height increase_offset detects the end of rising, NOT the start of lowering.
For lowering choose height decrease_onset. Raising and walking can overlap.
Unsigned speed alone cannot distinguish raising from lowering. An origin-distance
decrease can support approach but does NOT prove grasp or mesh contact; choose
labels/rationale that acknowledge visually inferred contact. Surface contact,
grasp force and support-plane contact are NOT available GT measurements.

ANATOMY:
Allowed body_parts: {', '.join(sorted(legacy.MAPPABLE_BODY_PARTS))}.
hand means palm/fingers; wrist means wrist region; elbow represents elbow/forearm.
Choose from visible evidence, not a hard-coded choice for an object category.
Criticality: 1 minor, 2 useful, 3 important, 4 failure-critical interaction.

AVAILABLE SIGNALS (definitions, not a request to estimate their numerical values):
{signal_guide(available)}
"""


def validate_plan(value: Any, available: set[str], duration: float) -> dict:
    if not isinstance(value, dict) or set(value) != {"schema", "actions"} or value["schema"] != SCHEMA:
        raise ValueError(f"expected schema={SCHEMA} and actions")
    actions = value["actions"]
    if not isinstance(actions, list) or not actions:
        raise ValueError("actions must be a nonempty list")
    previous = -1.0
    max_width = min(2.0, max(0.8, 0.4 * duration))
    for i, action in enumerate(actions):
        if not isinstance(action, dict) or set(action) != {
            "action", "body_parts", "criticality_level", "rationale", "start", "end"
        }:
            raise ValueError(f"action {i}: unexpected/missing fields (no numeric thresholds allowed)")
        if not isinstance(action["action"], str) or not action["action"].strip():
            raise ValueError(f"action {i}: missing name")
        parts = action["body_parts"]
        if (not isinstance(parts, list) or not parts or any(not isinstance(p, str) for p in parts)
                or set(parts) - legacy.MAPPABLE_BODY_PARTS or len(set(parts)) != len(parts)):
            raise ValueError(f"action {i}: invalid body_parts")
        if type(action["criticality_level"]) is not int or action["criticality_level"] not in legacy.CRITICALITY_MAPPING:
            raise ValueError(f"action {i}: criticality_level must be integer 1..4")
        if not isinstance(action["rationale"], str) or not action["rationale"].strip():
            raise ValueError(f"action {i}: missing visible rationale")
        midpoints = []
        for endpoint in ("start", "end"):
            rule = action[endpoint]
            if not isinstance(rule, dict) or set(rule) != {"time_range_s", "signal", "transition"}:
                raise ValueError(f"action {i}.{endpoint}: expected time_range_s, signal, transition ONLY")
            bounds = rule["time_range_s"]
            if (not isinstance(bounds, list) or len(bounds) != 2 or
                    any(type(t) not in (int, float) or not math.isfinite(t) for t in bounds)):
                raise ValueError(f"action {i}.{endpoint}: invalid time range")
            lo, hi = bounds
            if not 0 <= lo <= hi <= duration + 1e-5 or hi - lo > max_width + 1e-5:
                raise ValueError(f"action {i}.{endpoint}: out-of-range or overly broad search interval")
            transition, signal = rule["transition"], rule["signal"]
            if transition not in TRANSITIONS:
                raise ValueError(f"action {i}.{endpoint}: unsupported transition")
            if transition.startswith("clip_"):
                edge = 0.0 if transition == "clip_start" else duration
                valid_location = (transition == "clip_start" and i == 0 and endpoint == "start") or (
                    transition == "clip_end" and i == len(actions)-1 and endpoint == "end")
                if not valid_location or signal is not None or any(abs(t-edge) > 1e-5 for t in bounds):
                    raise ValueError(f"action {i}.{endpoint}: invalid clip-edge declaration")
            else:
                if signal not in available:
                    raise ValueError(f"action {i}.{endpoint}: unavailable signal {signal}")
                if _is_speed(signal) != transition.startswith("motion_"):
                    raise ValueError(f"action {i}.{endpoint}: speed signals use motion_onset/offset; other signals use increase/decrease")
                if signal in {"object_rotate", "body_travel_progress"} and transition.startswith("decrease"):
                    raise ValueError(f"action {i}.{endpoint}: accumulated signal cannot decrease")
            midpoints.append((lo + hi) / 2)
        if midpoints[0] >= midpoints[1] or midpoints[0] < previous:
            raise ValueError(f"action {i}: visual starts must be ordered and precede their ends")
        previous = midpoints[0]
    return value


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, width | 1)
    # Median rejects one-frame pose glitches before derivative estimation.
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(values, width//2, mode="edge"), width)
    median = np.median(windows, axis=1)
    return np.convolve(np.pad(median, width//2, mode="edge"), np.ones(width)/width, mode="valid")


def _mad(values: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def resolve_boundary(rule: dict, signals: dict[str, np.ndarray], fps: float,
                     config: DetectorConfig = DetectorConfig()) -> dict:
    n = len(next(iter(signals.values())))
    lo = max(0, int(math.ceil(rule["time_range_s"][0]*fps-1e-5)))
    hi = min(n-1, int(math.floor(rule["time_range_s"][1]*fps+1e-5)))
    trace = dict(rule=rule, status="unresolved", frame=None, search_frames=[lo, hi],
                 candidate_frames=[], threshold_source="local_quiet_baseline_and_noise")
    transition = rule["transition"]
    if transition.startswith("clip_"):
        trace.update(status="clip_censored", frame=0 if transition == "clip_start" else n-1,
                     threshold_source=None, reason="visual declaration; physical boundary outside observation")
        return trace
    if hi < lo:
        trace["reason"] = "visual interval contains no GT samples"
        return trace
    signal = rule["signal"]
    raw = np.asarray(signals[signal], dtype=float)
    width = max(1, int(round(config.smoothing_seconds*fps))) | 1
    smooth = _smooth(raw, width)
    is_speed = _is_speed(signal)
    feature = smooth if is_speed else np.gradient(smooth)*fps
    direction = -1 if transition.startswith("decrease") else 1
    feature = direction*feature
    floor = (config.angular_speed_floor if signal in {"object_rotate", "object_angular_speed"}
             else config.normalized_rate_floor if signal in {"object_progress", "body_travel_progress"}
             else config.linear_speed_floor)
    context = max(1, int(math.ceil(config.context_seconds*fps)))
    left, right = max(0, lo-context), min(n, hi+context+1)
    baseline_n = max(3, int(math.ceil(config.baseline_seconds*fps)))
    dwell = max(2, int(math.ceil(config.dwell_seconds*fps)))
    local = feature[left:right]
    trace.update(feature="unsigned_speed" if is_speed else "signed_derivative",
                 direction=direction, context_frames=[left, right-1], dwell_frames=dwell,
                 noise_floor=floor, smoothed_window_frames=width)
    if len(local) < baseline_n:
        trace["reason"] = "insufficient context for baseline"
        return trace
    # Choose the locally quietest contiguous segment, then verify it is quiet.
    # Selection cannot make a steadily moving clip into a quiet baseline.
    segments = np.lib.stride_tricks.sliding_window_view(local, baseline_n)
    scores = np.median(np.abs(segments), axis=1)
    j = int(np.argmin(scores))
    baseline = segments[j]
    center, sigma = float(np.median(baseline)), _mad(baseline)
    residual_noise = _mad(local - _smooth(local, width))
    quiet_limit = max(floor, config.noise_multiplier * residual_noise)
    trace.update(baseline_frames=[left+j, left+j+baseline_n-1], baseline_median=center,
                 baseline_mad_sigma=sigma, residual_mad_sigma=residual_noise, quiet_limit=quiet_limit)
    if float(np.median(np.abs(baseline))) > quiet_limit or sigma > quiet_limit:
        trace["reason"] = "no reliable local quiet baseline"
        return trace
    enter = max(floor, abs(center) + config.noise_multiplier*max(sigma, residual_noise))
    leave = config.exit_ratio*enter
    trace.update(enter_threshold=enter, exit_threshold=leave)
    # Hysteresis state has to be established by a sustained observed state.
    # No invented crossing at an already-moving first frame.
    state = None
    candidates = []
    want_onset = transition.endswith("_onset")
    for t in range(left, right-dwell+1):
        values = feature[t:t+dwell]
        active, quiet = bool(np.all(values >= enter)), bool(np.all(values <= leave))
        if state is None:
            if active:
                state = True
            elif quiet:
                state = False
            continue
        if not state and active:
            if want_onset and lo <= t <= hi:
                candidates.append(t)
            state = True
        elif state and quiet:
            if not want_onset and lo <= t <= hi:
                candidates.append(t)
            state = False
    trace["candidate_frames"] = candidates
    if not candidates:
        trace["reason"] = "no sustained transition inside the visual interval"
        return trace
    # Only genuine transitions may compete; closeness never rescues a nonmatch.
    midpoint = sum(rule["time_range_s"])*fps/2
    selected = min(candidates, key=lambda t: (abs(t-midpoint), t))
    trace.update(status="gt_resolved", frame=int(selected),
                 selection="valid transition nearest visual interval midpoint",
                 feature_at_frame=float(feature[selected]),
                 multiple_candidates=len(candidates)>1)
    return trace


def resolve_plan(plan: dict, signals: dict[str, np.ndarray], fps: float,
                 config: DetectorConfig = DetectorConfig()) -> dict:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    n = len(next(iter(signals.values())))
    if n < 3 or any(np.asarray(v).shape != (n,) or not np.isfinite(v).all() for v in signals.values()):
        raise ValueError("signals must be finite, aligned 1D arrays with >=3 samples")
    validate_plan(plan, set(signals) & legacy.DYNAMIC_SIGNALS, (n-1)/fps)
    actions, issues, events = [], [], []
    previous_start = -1
    for i, action in enumerate(plan["actions"]):
        boundaries = {end: resolve_boundary(action[end], signals, fps, config) for end in ("start", "end")}
        start, end = boundaries["start"]["frame"], boundaries["end"]["frame"]
        item = dict(action_index=i, action=action["action"], boundaries=boundaries)
        actions.append(item)
        for endpoint, trace in boundaries.items():
            if trace["status"] == "unresolved":
                issues.append(dict(action_index=i, action=action["action"], endpoint=endpoint, reason=trace["reason"]))
        if start is None or end is None:
            continue
        if start >= end or start < previous_start:
            issues.append(dict(action_index=i, action=action["action"], endpoint="window",
                               reason="nonpositive duration or reversed action-start order", frames=[start,end]))
            continue
        previous_start = start
        events.append(dict(
            event=f"{i:02d}_{action['action']}", action=action["action"], body_parts=action["body_parts"],
            criticality_level=action["criticality_level"],
            criticality=legacy.CRITICALITY_MAPPING[action["criticality_level"]],
            rationale=action["rationale"], windows=[dict(start_frame=start, end_frame=end, trigger_frame=start)],
            boundary_evidence=boundaries,
        ))
    accepted = not issues and len(events) == len(plan["actions"])
    return dict(schema=SCHEMA, fps=fps, source_frames=n, status="resolved" if accepted else "unresolved",
                detector_config=asdict(config), detector_calibration="experimental defaults; not fitted to these tasks",
                actions=actions, issues=issues,
                # Partial events are diagnostic only, never an accepted pruned plan.
                resolved_events=events, semantic_mode="bounded_visual_intervals_gt_changes")


def execute_bounded_plan(plan: dict, signals: dict[str, np.ndarray], fps: float = 30.0) -> dict:
    result = resolve_plan(plan, signals, fps)
    if result["status"] != "resolved":
        raise ValueError(f"unresolved bounded plan: {result['issues']}")
    return dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"],
                detector_config=result["detector_config"])


def _write(path: Path, value: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def generate_bounded_plan(*, video: Path, bundle_file: Path, output: Path,
                          audit_dir: Path | None = None, max_repairs: int = 3,
                          call: Callable | None = None) -> dict:
    """Save every candidate; only emit semantic_plan.json if the whole plan resolves."""
    if max_repairs < 0:
        raise ValueError("max_repairs must be nonnegative")
    audit = audit_dir or output.parent / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    if output.exists() or list(audit.glob("attempt_*.response.txt")):
        raise FileExistsError("use a new output/audit directory; existing attempts must remain auditable")
    legacy.load_env_file(Path(".env"))
    signals, _, fps = legacy.load_retargeting_bundle_signals(bundle_file)
    n = len(next(iter(signals.values())))
    available = set(signals) & legacy.DYNAMIC_SIGNALS
    base_prompt = build_prompt(available, (n-1)/fps)
    (audit / "base_prompt.txt").write_text(base_prompt, encoding="utf-8")
    video_bytes = video.read_bytes()
    content = [{"type":"video_url", "video_url":{
        "url":"data:video/mp4;base64,"+base64.b64encode(video_bytes).decode("ascii")}}]
    metadata = dict(model=os.environ.get("OPENAI_MODEL"), prompt_version=PROMPT_VERSION,
                    source_video=str(video), source_video_sha256=hashlib.sha256(video_bytes).hexdigest(),
                    source_bundle=str(bundle_file), source_bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
                    source_frames=n, fps=fps, duration_s=(n-1)/fps, visual_source_kind="full_mp4_video",
                    client_frame_sampling=False, server_frame_sampling="unknown backend defaults",
                    base_prompt_sha256=hashlib.sha256(base_prompt.encode()).hexdigest(),
                    temperature=0, seed=0, threshold_policy="local quiet baseline and GT noise; no VLM numbers")
    _write(audit / "metadata.json", metadata)
    previous, feedback = None, []
    request = call or legacy.call_vlm
    for attempt in range(max_repairs+1):
        prompt = base_prompt
        if attempt:
            prompt += ("\nPrevious candidate:\n"+json.dumps(previous, ensure_ascii=False)+
                       "\nValidation feedback:\n"+json.dumps(feedback, ensure_ascii=False)+
                       "\nReview the full video and correct timing/signal/transition selections. Return the complete JSON. "
                       "Do not delete visible actions or misuse clip edges to make validation pass. "
                       "Do not invent new signal names or numeric thresholds. A truly unsupported boundary may remain unresolved.")
        prefix = audit / f"attempt_{attempt}"
        prefix.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
        raw = request(content, prompt)
        prefix.with_suffix(".response.txt").write_text(raw, encoding="utf-8")
        try:
            previous = legacy._extract_json_value(raw)
            plan = validate_plan(previous, available, (n-1)/fps)
            _write(prefix.with_suffix(".plan.json"), plan)
            result = resolve_plan(plan, signals, fps)
            _write(prefix.with_suffix(".resolution.json"), result)
            feedback = result["issues"]
            if result["status"] != "resolved":
                continue
            semantic = dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"],
                            detector_config=result["detector_config"],
                            generation_metadata=dict(metadata, vlm_attempt_count=attempt+1,
                                                     detector_calibration=result["detector_calibration"],
                                                     visual_review="pending"))
            legacy.validate_semantic_keyframe_json(semantic)
            _write(output.with_name(output.stem+".event_plan.json"), plan)
            _write(output, semantic)
            return semantic
        except (ValueError, TypeError, KeyError) as exc:
            feedback = [dict(reason=str(exc))]
            _write(prefix.with_suffix(".validation.json"), feedback)
    raise ValueError(f"whole plan remains unresolved after {max_repairs+1} requests: {feedback}")
