"""VLM-written signed change ratios, normalized by full-clip signal amplitude.

No smoothing, stationary-baseline estimation, physical speed floors, dwell or
hysteresis. The only event thresholds are the dimensionless ratios written by
the VLM. Crossings retain direction; unsupported rules never get a fallback.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import pipeline as legacy

SCHEMA = "holosoma.range_delta_plan.v1"
PROMPT_VERSION = "range_normalized_signed_delta_v1"
POLICY = {
    "ratio": "(s[t] - s[reference_frame]) / (max(s_whole_clip) - min(s_whole_clip))",
    "references": ["clip_start", "range_start", "action_start"],
    "crossings": {"up": "r[t-1] < ratio <= r[t]", "down": "r[t-1] > ratio >= r[t]"},
    "selection": "earliest matching crossing within visual range, after reference frame",
    "zero_span": "unresolved; no artificial epsilon denominator",
    "signal_filters": "none",
    "unmatched": "unresolved; no nearest-value fallback",
}


def build_prompt(available: set[str], duration: float) -> str:
    signals = legacy.build_dynamic_signal_guide(available).replace(
        "Increasing quantile crossings", "Increasing ratio crossings")
    return f"""Watch the ENTIRE supplied video, from 0 to {duration:.6f} seconds.
Describe the actual visible human-object interaction phases, the involved body
regions, and executable start/end rules. Do not assume a lift/carry/place script.
Do not invent actions after the clip ends. Retain meaningful holding/carrying
phases if visible. Simultaneous phases may overlap; repeated actions may recur.

Use SIGNED CHANGE RATIOS of GT signals, NOT physical scalar thresholds or quantiles:
  D = max(s over the WHOLE clip) - min(s over the WHOLE clip)
  r[t] = (s[t] - s[reference_frame]) / D
The same D is used for every boundary using that signal. +0.20 means an INCREASE
of 20% of the signal's whole-clip amplitude since the chosen reference; -0.20
means a DECREASE of 20% of that amplitude. This is NOT a 20% increase relative
to the initial value, NOT a percentile rank, and NOT a fraction of elapsed time.
Initial signal value can be zero. If D=0 the rule is unresolved; never divide by
an invented epsilon. Small jitter can have D>0: use visible evidence to avoid
describing an almost-flat signal as meaningful motion just because it is normalized.

You choose the signal, reference, signed ratio, crossing direction and approximate
visual TIME RANGE for each boundary. The program does not fit any noise thresholds,
stationary baselines, minimum speeds, dwell times, smoothing windows or hysteresis.

REFERENCE (one of these exact strings):
- clip_start: reference_frame=0. Useful for change relative to the whole clip's initial pose.
- range_start: reference_frame=first GT frame inside THIS boundary's time_range_s.
  Useful for a local rise/drop. Its ratio starts at zero at this local reference.
- action_start: reference_frame=the resolved start of THIS action. Allowed ONLY
  for an end rule. Useful for how much changed during the action.
All candidate crossings must be AFTER the reference frame. The reference is fixed
throughout the search, not reset at every frame. Choose it before the change you
want to measure. Changing a time range also changes a range_start reference.

CROSSING DIRECTION, independent of the sign of the ratio:
- up: earliest r[t-1] < ratio AND r[t] >= ratio in the visual time range.
- down: earliest r[t-1] > ratio AND r[t] <= ratio in the visual time range.
A positive ratio can be crossed DOWN while the signal is still above its reference.
For instance, [0,1,2,3,2,1,0] has D=3 and reference=first sample. ratio=+0.5
is crossed UP at index 2, DOWN at index 5. These numbers are an arithmetic example,
not an action template. A negative ratio requires the signal to fall BELOW its
reference. A down crossing of speed means slowing down, NOT moving downwards.
To distinguish upward/downward object motion use object_height with up/down.
Unsigned angular speed/rotation cannot reveal clockwise versus counterclockwise.
Do not introduce minimum/maximum EVENT rules: min/max above only define D.

OUTPUT: one JSON object, no prose/code/Markdown. All numbers must be JSON numbers.
{{"schema":"{SCHEMA}","actions":[{{
  "action":"<short descriptive action name>",
  "body_parts":["<allowed body region>"],
  "criticality_level":3,
  "rationale":"<visible evidence for this phase, anatomy, and selected signals>",
  "start":{{"time_range_s":["<earliest plausible time>","<latest plausible time>"],
             "signal":"<available signal>","reference":"<reference>",
             "ratio":"<signed number between -1 and 1>","crossing":"<up or down>"}},
  "end":{{"time_range_s":["<earliest plausible time>","<latest plausible time>"],
           "signal":"<available signal>","reference":"<reference>",
           "ratio":"<signed number between -1 and 1>","crossing":"<up or down>"}}
}}]}}
Every time range lies within [0,{duration:.6f}] and has low <= high. Estimate a
coarse neighborhood around the visible boundary. Do not set every range to the
whole clip or alter action meaning just to find any signal crossing. Start range
midpoint precedes end range midpoint; action starts are in chronological order.
An end searches strictly AFTER its action start. There is no forced clipping to
the next action, and no nearest-value fallback when a crossing does not exist.

VIDEO EDGES have a SIMPLE STRING syntax, instead of a signal-rule object:
- If the first phase is already underway in the first frame, write "start":"clip_start".
- If the last phase continues beyond the last frame, write "end":"clip_end".
These are visually declared censored observations, not GT-confirmed motion changes.
Use the exact string, not an object containing nulls, times, or a fake signal.

ANATOMY: body_parts may contain ONLY {', '.join(sorted(legacy.MAPPABLE_BODY_PARTS))}.
The object itself is NOT a body part. Hand=palm/fingers; wrist=wrist region;
elbow includes forearm support. Select regions from visible evidence; do not
always pick hands or include parent/nearby joints automatically. Criticality:
1 minor, 2 useful, 3 important, 4 failure-critical. No fixed event count is required.

SIGNAL DEFINITIONS below describe the raw GT quantities, not values you must
estimate in meters or m/s. Use ONLY the dimensionless ratio output above. Distances
to the object origin do not measure mesh contact or grasp force. Ground clearance
is not measured by object_height. Body travel and cumulative rotation only increase.
{signals}
"""


def validation_issues(plan, available: set[str], duration: float) -> list[dict]:
    """Report all independent schema errors in one repair, not one at a time."""
    issues = []

    def add(field, reason):
        issues.append({"field": field, "reason": reason})

    if not isinstance(plan, dict) or set(plan) != {"schema", "actions"} or plan.get("schema") != SCHEMA:
        return [{"field":"plan", "reason":f"expected schema={SCHEMA} and actions only"}]
    actions = plan["actions"]
    if not isinstance(actions, list) or not actions:
        return [{"field":"actions", "reason":"must be a nonempty list"}]
    previous_midpoint = -1.0
    for i, action in enumerate(actions):
        prefix = f"actions[{i}]"
        if not isinstance(action, dict) or set(action) != {
                "action", "body_parts", "criticality_level", "rationale", "start", "end"}:
            add(prefix, "expected action/body_parts/criticality_level/rationale/start/end only")
            continue
        for key in ("action", "rationale"):
            if not isinstance(action[key], str) or not action[key].strip():
                add(prefix+"."+key, "must be a nonempty string")
        parts = action["body_parts"]
        if not isinstance(parts, list) or not parts or any(not isinstance(p, str) for p in parts):
            add(prefix+".body_parts", "must be a nonempty list of allowed body regions")
        elif set(parts)-legacy.MAPPABLE_BODY_PARTS or len(set(parts)) != len(parts):
            add(prefix+".body_parts", f"invalid or duplicate body regions: {parts}; object is not a body region")
        if type(action["criticality_level"]) is not int or action["criticality_level"] not in legacy.CRITICALITY_MAPPING:
            add(prefix+".criticality_level", "must be an integer 1..4")
        mids = {}
        for endpoint in ("start", "end"):
            field = prefix+"."+endpoint
            rule = action[endpoint]
            if isinstance(rule, str):
                if rule == "clip_start" and endpoint == "start" and i == 0:
                    mids[endpoint] = 0.0
                elif rule == "clip_end" and endpoint == "end" and i == len(actions)-1:
                    mids[endpoint] = duration
                else:
                    add(field, 'edge strings only: first start="clip_start" or last end="clip_end"')
                continue
            if not isinstance(rule, dict) or set(rule) != {"time_range_s", "signal", "reference", "ratio", "crossing"}:
                add(field, "expected time_range_s/signal/reference/ratio/crossing, or an allowed edge STRING")
                continue
            times = rule["time_range_s"]
            if (not isinstance(times, list) or len(times) != 2 or
                    any(type(t) not in (float,int) or not math.isfinite(t) for t in times)):
                add(field+".time_range_s", "must contain two finite numbers")
            elif not 0 <= times[0] <= times[1] <= duration+1e-5:
                add(field+".time_range_s", f"must be ordered inside [0,{duration:.6f}]")
            else:
                mids[endpoint] = sum(times)/2
            signal = rule["signal"]
            if not isinstance(signal, str) or signal not in available:
                add(field+".signal", f"unavailable signal: {signal}")
            reference = rule["reference"]
            if reference not in ("clip_start", "range_start", "action_start"):
                add(field+".reference", "must be clip_start, range_start or action_start")
            elif reference == "action_start" and endpoint == "start":
                add(field+".reference", "action_start is circular for a start rule; use clip_start/range_start")
            ratio = rule["ratio"]
            if type(ratio) not in (int,float) or not math.isfinite(ratio) or not -1 <= ratio <= 1:
                add(field+".ratio", "must be a signed number in [-1,1], not a percentage string or quantile")
            if rule["crossing"] not in ("up", "down"):
                add(field+".crossing", "must be up or down")
            if signal in ("object_rotate", "body_travel_progress") and (
                    rule["crossing"] == "down" or (type(ratio) in (int,float) and ratio <= 0)):
                add(field, "nondecreasing cumulative signal requires a positive ratio and up crossing")
        if "start" in mids:
            if mids["start"] < previous_midpoint:
                add(prefix+".start", "visual start midpoints must be chronological")
            previous_midpoint = mids["start"]
        if len(mids) == 2 and mids["start"] >= mids["end"]:
            add(prefix, "visual start must precede visual end")
    return issues


def resolve_boundary(rule, signals, fps, *, action_start=None, minimum=0):
    n = len(next(iter(signals.values())))
    trace = dict(rule=rule, status="unresolved", frame=None, candidate_frames=[])
    if isinstance(rule, str):
        trace.update(status="clip_censored", frame=0 if rule == "clip_start" else n-1,
                     reason="visual edge declaration, not a measured transition")
        return trace
    lo = max(0, int(math.ceil(rule["time_range_s"][0]*fps-1e-5)))
    hi = min(n-1, int(math.floor(rule["time_range_s"][1]*fps+1e-5)))
    reference = {"clip_start":0, "range_start":lo, "action_start":action_start}[rule["reference"]]
    trace.update(search_frames=[lo,hi], reference_frame=reference)
    if reference is None:
        trace["reason"] = "action-start reference unresolved"
        return trace
    if not 0 <= reference < n or hi < lo:
        trace["reason"] = "time range/reference contains no GT samples"
        return trace
    raw = np.asarray(signals[rule["signal"]], dtype=float)
    low, high = float(np.min(raw)), float(np.max(raw))
    span = high-low
    trace.update(signal_min=low, signal_max=high, signal_span=span, reference_value=float(raw[reference]),
                 ratio_threshold=rule["ratio"], crossing=rule["crossing"])
    if span == 0:
        trace["reason"] = "constant signal: full-clip amplitude is zero"
        return trace
    ratios = (raw-raw[reference])/span
    threshold = rule["ratio"]
    if rule["crossing"] == "up":
        matches = (ratios[:-1] < threshold) & (ratios[1:] >= threshold)
    else:
        matches = (ratios[:-1] > threshold) & (ratios[1:] <= threshold)
    first = max(lo, reference+1, minimum, 1)
    candidates = np.flatnonzero(matches)+1
    candidates = candidates[(candidates >= first) & (candidates <= hi)]
    trace.update(effective_search_frames=[first,hi], candidate_frames=candidates.tolist(),
                 raw_threshold=float(raw[reference]+threshold*span))
    if first <= hi:
        trace["observed_ratio_range"] = [float(np.min(ratios[first:hi+1])), float(np.max(ratios[first:hi+1]))]
    if not len(candidates):
        trace["reason"] = "no directional ratio crossing inside the visual interval"
        return trace
    frame = int(candidates[0])
    trace.update(status="gt_resolved", frame=frame, ratio_before=float(ratios[frame-1]),
                 ratio_at_frame=float(ratios[frame]), multiple_candidates=len(candidates)>1,
                 selection="earliest actual crossing in the declared range")
    return trace


def resolve_plan(plan, signals, fps):
    if not math.isfinite(fps) or fps <= 0 or not signals:
        raise ValueError("finite positive fps and nonempty signals required")
    n = len(next(iter(signals.values())))
    if n < 2 or any(np.asarray(v).shape != (n,) or not np.isfinite(v).all() for v in signals.values()):
        raise ValueError("finite aligned 1D signals with at least two samples required")
    errors = validation_issues(plan, set(signals)&legacy.DYNAMIC_SIGNALS, (n-1)/fps)
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    actions, issues, events = [], [], []
    previous_start = -1
    for i, action in enumerate(plan["actions"]):
        start = resolve_boundary(action["start"], signals, fps, minimum=max(0,previous_start))
        end = resolve_boundary(action["end"], signals, fps, action_start=start["frame"],
                               minimum=0 if start["frame"] is None else start["frame"]+1)
        boundaries = {"start":start, "end":end}
        actions.append(dict(action_index=i, action=action["action"], boundaries=boundaries))
        for endpoint, bound in boundaries.items():
            if bound["status"] == "unresolved":
                issues.append(dict(action_index=i, action=action["action"], endpoint=endpoint, reason=bound["reason"]))
        a, z = start["frame"], end["frame"]
        if a is None or z is None:
            continue
        if a >= z or a < previous_start:
            issues.append(dict(action_index=i, endpoint="window", reason="nonpositive duration or reversed start order"))
            continue
        previous_start = a
        events.append(dict(event=f"{i:02d}_{action['action']}", action=action["action"],
                           body_parts=action["body_parts"], criticality_level=action["criticality_level"],
                           criticality=legacy.CRITICALITY_MAPPING[action["criticality_level"]],
                           rationale=action["rationale"], boundary_evidence=boundaries,
                           windows=[dict(start_frame=a, end_frame=z, trigger_frame=a)]))
    return dict(schema=SCHEMA, status="resolved" if not issues else "unresolved", fps=fps, source_frames=n,
                semantic_mode="vlm_range_normalized_signed_changes", normalization_policy=POLICY,
                actions=actions, issues=issues, resolved_events=events)


def execute_range_delta_plan(plan, signals, fps=30.0):
    result = resolve_plan(plan, signals, fps)
    if result["status"] != "resolved":
        raise ValueError(f"unresolved range-delta plan: {result['issues']}")
    return dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"],
                normalization_policy=POLICY)


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def generate_range_delta_plan(*, video: Path, bundle_file: Path, output: Path,
                              audit_dir: Path | None = None, max_repairs=3, call=None):
    if not 0 <= max_repairs <= 3:
        raise ValueError("max_repairs must be 0..3")
    audit = audit_dir or output.parent/"audit"
    audit.mkdir(parents=True, exist_ok=True)
    if output.exists() or list(audit.glob("attempt_*.response.txt")):
        raise FileExistsError("use a fresh output/audit directory")
    legacy.load_env_file(Path(".env"))
    signals, _, fps = legacy.load_retargeting_bundle_signals(bundle_file)
    n = len(next(iter(signals.values())))
    available = set(signals)&legacy.DYNAMIC_SIGNALS
    prompt_base = build_prompt(available, (n-1)/fps)
    (audit/"base_prompt.txt").write_text(prompt_base)
    video_bytes = video.read_bytes()
    content = [{"type":"video_url", "video_url":{
        "url":"data:video/mp4;base64,"+base64.b64encode(video_bytes).decode("ascii")}}]
    metadata = dict(model=os.environ.get("OPENAI_MODEL"), prompt_version=PROMPT_VERSION,
                    source_video=str(video), source_video_sha256=hashlib.sha256(video_bytes).hexdigest(),
                    source_bundle=str(bundle_file), source_bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
                    source_frames=n, fps=fps, duration_s=(n-1)/fps, visual_source_kind="full_mp4_video",
                    client_frame_sampling=False, server_frame_sampling="unknown backend defaults",
                    temperature=0, seed=0, normalization_policy=POLICY,
                    base_prompt_sha256=hashlib.sha256(prompt_base.encode()).hexdigest())
    _write(audit/"metadata.json", metadata)
    previous, feedback = None, []
    request = call or legacy.call_vlm
    for attempt in range(max_repairs+1):
        prompt = prompt_base
        if attempt:
            prompt += ("\nPrevious candidate:\n"+json.dumps(previous, ensure_ascii=False)+
                       "\nALL validation issues:\n"+json.dumps(feedback, ensure_ascii=False)+
                       "\nReview the whole video and correct each affected time/reference/signal/ratio/direction. "
                       "Return the complete plan. Do not delete observed phases just to pass. "
                       "Do not use physical thresholds, quantiles, extrema, or fake clip edges.")
        prefix = audit/f"attempt_{attempt}"
        prefix.with_suffix(".prompt.txt").write_text(prompt)
        raw = request(content, prompt)
        prefix.with_suffix(".response.txt").write_text(raw)
        try:
            previous = legacy._extract_json_value(raw)
            feedback = validation_issues(previous, available, (n-1)/fps)
            if feedback:
                _write(prefix.with_suffix(".validation.json"), feedback)
                continue
            _write(prefix.with_suffix(".plan.json"), previous)
            result = resolve_plan(previous, signals, fps)
            _write(prefix.with_suffix(".resolution.json"), result)
            feedback = result["issues"]
            if feedback:
                continue
            semantic = dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"],
                            normalization_policy=POLICY,
                            generation_metadata=dict(metadata, vlm_attempt_count=attempt+1, visual_review="pending"))
            legacy.validate_semantic_keyframe_json(semantic)
            _write(output.with_name(output.stem+".event_plan.json"), previous)
            _write(output, semantic)
            return semantic
        except (ValueError, TypeError, KeyError) as exc:
            feedback = [dict(reason=str(exc))]
            _write(prefix.with_suffix(".validation.json"), feedback)
    raise ValueError(f"whole plan remains unresolved after {max_repairs+1} requests: {feedback}")
