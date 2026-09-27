"""Locate two keyframes using separate adjacent-frame delta predicates.

Only the selected start and end frames must satisfy their respective rules.
Neither the VLM nor the executor selects a continuously satisfying interval.
The older adjacent_delta module remains readable for archived experiments.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from . import adjacent_delta as shared
from . import pipeline as legacy

LEGACY_SCHEMA = "holosoma.adjacent_delta_boundary_plan.v1"
SCHEMA = "holosoma.adjacent_delta_boundary_plan.v2"
PROMPT_VERSION = "adjacent_delta_compact_v2"
SOURCE_DEPENDENCIES = (shared,)
POLICY = {
    "delta": "s[t] - s[t-1], t=1..N-1; frame zero has no predecessor",
    "ratio": "delta[t] / (max(s_whole_clip) - min(s_whole_clip))",
    "conditions": {"increasing": "d[t] >= threshold > 0", "decreasing": "d[t] <= -threshold < 0",
                   "steady": "abs(d[t]) <= threshold"},
    "start": "first matching current frame t strictly after the preceding action's start",
    "end": "first matching current frame t at or after this action's start",
    "interior": "unconstrained; start and end predicates need not hold between the two keyframes",
    "ordering": "starts advance; endpoints are not clipped to other actions; overlaps and point events allowed",
    "zero_span": "unresolved; no artificial denominator or fabricated ratio",
    "filters_or_dwell": "none; no runs, duration requirement, or delta-threshold crossing requirement",
    "fps_policy": "native GT rate, no resampling or multiplication by fps; thresholds are per frame",
    "unmatched": "unresolved; no nearest-value fallback, boundary replacement, or action pruning",
}


def compact_signal_guide(available):
    """Factor repeated joint definitions while preserving the same vocabulary."""
    suffixes = ("height", "speed", "object_distance")
    parts = sorted(part for part in legacy.MAPPABLE_BODY_PARTS
                   if all(f"{part}_{suffix}" in available for suffix in suffixes))
    grouped = {f"{part}_{suffix}" for part in parts for suffix in suffixes}
    descriptions = {
        "body_travel_progress": "cumulative normalized mean joint travel; nondecreasing, not time",
        "hand_mean_speed": "mean unsigned speed of both hands",
        "max_hand_object_distance": "larger of the two hand-to-object-origin distances",
        "min_hand_object_distance": "smaller of the two hand-to-object-origin distances",
        "object_angular_speed": "unsigned angular speed",
        "object_displacement": "distance from initial object position, not path length",
        "object_goal_distance": "distance to final object position",
        "object_height": "object-origin world Z minus first-frame Z, not ground clearance",
        "object_progress": "clipped displacement projection toward final object position, not time",
        "object_rotate": "cumulative unsigned rotation; nondecreasing",
        "object_speed": "unsigned 3D speed",
        "object_vertical_speed": "absolute vertical speed, positive for both upward and downward motion",
    }
    lines = []
    if parts:
        lines += [f"For each part in [{', '.join(parts)}], these three exact signal names exist:",
                  "<part>_height = joint world Z; <part>_speed = unsigned 3D speed;",
                  "<part>_object_distance = joint-to-object-ORIGIN distance (not surface contact)."]
    for name in sorted(set(available)-grouped):
        if name in descriptions:
            lines.append(f"{name}: {descriptions[name]}.")
        else:
            lines.append(shared.signal_guide({name}))
    return "\n".join(lines)


def build_prompt(available: set[str], frame_count: int, fps: float) -> str:
    return f"""Watch the full video ({frame_count} GT frames, {fps:g} fps). Identify visible actions
in start order and their important body regions. Assess the visual evidence and
boundary rules internally; output ONLY the final JSON, without explanations.
Preserve distinct visible actions/repetitions; do not assume an action sequence.

ANATOMY: choose each exact region from visible evidence, not a default body list
or a nearby/parent region. Hand=palm/fingers; wrist=wrist; elbow includes forearm
support/clamping; waist=trunk. Check the evidence for each selected region internally.
Allowed body_parts: {', '.join(sorted(legacy.MAPPABLE_BODY_PARTS))}. Object is not a body part.

Write separate START FRAME and END FRAME predicates. Each has one signal,
condition, and numeric threshold. The comparison is always adjacent GT samples:
D=max(s over WHOLE clip)-min(s over WHOLE clip); d[t]=(s[t]-s[t-1])/D.
D is raw signal amplitude. Frame 0 has no delta; D=0 cannot resolve.
increasing: d[t]>=threshold>0; decreasing: d[t]<=-threshold<0;
steady: abs(d[t])<=threshold, threshold>=0. Threshold is a fraction in [0,1];
0.02 means 2% of full amplitude in ONE frame step, not cumulative change or quantile.
Start/end may use different signals and thresholds.

The program selects the first current frame t satisfying the start rule after
the previous action's start, then the first frame t satisfying the end rule at
or after this start. The frames BETWEEN them need not satisfy either predicate.
It does not select a continuous true run or add another crossing test on d[t-1].
Start=end is allowed; short duration alone is not an error. Overlaps are allowed;
ends are not clipped to the next start. No smoothing, dwell or nearest-frame fallback.
Do not output frame numbers, times, intervals, reference choices, or extra fields.

SIGNALS: height delta distinguishes rise/fall; distance delta approach/separation.
Speed delta is acceleration/deceleration, not direction or proof of stopping.
Steady height alone does not distinguish holding/carrying/resting. Origin distance
is not contact. Tiny raw spans can amplify jitter. Use visible evidence.
{compact_signal_guide(available)}

Return exactly this structure; placeholders are types, not example answers:
{{"schema":"{SCHEMA}","actions":[{{"action":"<action name>",
"body_parts":["<allowed region>"],
"start":{{"signal":"<signal>","condition":"<condition>","threshold":"<number>"}},
"end":{{"signal":"<signal>","condition":"<condition>","threshold":"<number>"}}}}]}}
Replace thresholds with JSON numbers; no prose or Markdown.
"""


def validation_issues(plan, available):
    if not isinstance(plan, dict) or set(plan) != {"schema", "actions"} or plan.get("schema") not in (SCHEMA, LEGACY_SCHEMA):
        return [dict(field="plan", reason=f"expected only schema={SCHEMA} and actions")]
    if not isinstance(plan["actions"], list) or not plan["actions"]:
        return [dict(field="actions", reason="nonempty list required")]
    issues = []
    for i, action in enumerate(plan["actions"]):
        field = f"actions[{i}]"
        fields = {"action", "body_parts", "start", "end"}
        if plan["schema"] == LEGACY_SCHEMA:
            fields.add("rationale")
        if not isinstance(action, dict) or set(action) != fields:
            issues.append(dict(field=field, reason=f"only {','.join(sorted(fields))}; separate boundary rules required"))
            continue
        for key in (("action", "rationale") if plan["schema"] == LEGACY_SCHEMA else ("action",)):
            if not isinstance(action[key], str) or not action[key].strip():
                issues.append(dict(field=field+"."+key, reason="nonempty string required"))
        parts = action["body_parts"]
        if (not isinstance(parts, list) or not parts or any(not isinstance(x, str) for x in parts)
                or len(parts) != len(set(parts)) or set(parts)-legacy.MAPPABLE_BODY_PARTS):
            issues.append(dict(field=field+".body_parts", reason="use unique allowed body regions only"))
        for endpoint in ("start", "end"):
            key = field+"."+endpoint
            rule = action[endpoint]
            if not isinstance(rule, dict) or set(rule) != {"signal", "condition", "threshold"}:
                issues.append(dict(field=key, reason="only signal/condition/threshold; no frame/time/reference/duration"))
                continue
            signal, condition, threshold = rule["signal"], rule["condition"], rule["threshold"]
            if not isinstance(signal, str) or signal not in available:
                issues.append(dict(field=key+".signal", reason="unavailable signal"))
            if condition not in ("increasing", "decreasing", "steady"):
                issues.append(dict(field=key+".condition", reason="must be increasing, decreasing or steady"))
            if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                issues.append(dict(field=key+".threshold", reason="one finite JSON number in [0,1] required"))
            elif condition != "steady" and threshold == 0:
                issues.append(dict(field=key+".threshold", reason="increasing/decreasing require a positive magnitude"))
            if signal in ("object_rotate", "body_travel_progress") and condition == "decreasing":
                issues.append(dict(field=key, reason="cumulative nondecreasing signals cannot decrease"))
    return issues


def resolve_boundary(rule, signals, minimum=1):
    """Return the first satisfying current frame; no run extraction or crossing."""
    raw = np.asarray(signals[rule["signal"]], dtype=float)
    ratios, span = shared.delta_ratios(raw)
    trace = dict(**rule, status="unresolved", frame=None, minimum_frame=max(1, int(minimum)),
                 signal_min=float(np.min(raw)), signal_max=float(np.max(raw)), signal_span=span,
                 raw_step_threshold=float(rule["threshold"]*span), candidate_frames=[], eligible_frames=[])
    if ratios is None:
        trace["reason"] = "constant signal: normalized adjacent difference is undefined"
        return trace
    tau = rule["threshold"]
    mask = (ratios >= tau if rule["condition"] == "increasing" else
            ratios <= -tau if rule["condition"] == "decreasing" else np.abs(ratios) <= tau)
    mask[0] = False
    candidates = np.flatnonzero(mask)
    eligible = candidates[candidates >= trace["minimum_frame"]]
    trace.update(candidate_frames=candidates.tolist(), eligible_frames=eligible.tolist(),
                 observed_delta_ratio_range=[float(np.min(ratios[1:])), float(np.max(ratios[1:]))])
    if not len(eligible):
        trace["reason"] = "no matching frame at or after chronological lower bound"
        return trace
    t = int(eligible[0])
    trace.update(status="gt_resolved", frame=t, adjacent_frames=[t-1, t], ratio=float(ratios[t]),
                 previous_value=float(raw[t-1]), current_value=float(raw[t]), multiple_candidates=len(eligible)>1)
    return trace


def resolve_plan(plan, signals, fps):
    if not signals or not math.isfinite(fps) or fps <= 0:
        raise ValueError("nonempty signals and finite positive fps required")
    n = len(next(iter(signals.values())))
    if n < 2 or any(np.asarray(v).shape != (n,) or not np.isfinite(v).all() for v in signals.values()):
        raise ValueError("finite aligned 1D signals required")
    errors = validation_issues(plan, set(signals)&legacy.DYNAMIC_SIGNALS)
    if errors:
        raise ValueError(json.dumps(errors, ensure_ascii=False))
    actions, issues, events = [], [], []
    previous_start = 0
    start_order_blocked = False
    for i, action in enumerate(plan["actions"]):
        start = resolve_boundary(action["start"], signals, previous_start+1)
        if start_order_blocked:
            start["diagnostic_candidate_frame"] = start["frame"]
            start.update(status="blocked_by_previous_start", frame=None,
                         reason="earlier start unresolved; chronological lower bound unknown")
        end = resolve_boundary(action["end"], signals, start["frame"] if start["frame"] is not None else 1)
        if start["frame"] is None:
            start_order_blocked = True
            end["diagnostic_candidate_frame"] = end["frame"]
            end.update(status="blocked_by_start", frame=None, reason="this action's start is unresolved")
        else:
            previous_start = start["frame"]
        boundaries = dict(start=start, end=end)
        actions.append(dict(action_index=i, action=action["action"], boundaries=boundaries))
        for endpoint, bound in boundaries.items():
            if bound["status"] != "gt_resolved":
                issues.append(dict(action_index=i, action=action["action"], endpoint=endpoint, reason=bound["reason"]))
        a, b = start["frame"], end["frame"]
        if a is None or b is None:
            continue
        event = dict(event=f"{i:02d}_{action['action']}", action=action["action"], body_parts=action["body_parts"],
                     boundary_evidence=boundaries, windows=[dict(start_frame=a, end_frame=b, trigger_frame=a)])
        if plan["schema"] == LEGACY_SCHEMA:
            # Historical replay only; new plans have no explanation or weight fields.
            event.update(criticality_level=3, criticality=legacy.CRITICALITY_MAPPING[3],
                         criticality_source="uniform experiment placeholder; not VLM selected", rationale=action["rationale"])
        events.append(event)
    return dict(schema=plan["schema"], status="resolved" if not issues else "unresolved", fps=fps, source_frames=n,
                semantic_mode=("vlm_adjacent_delta_separate_keyframes" if plan["schema"] == LEGACY_SCHEMA
                               else "vlm_adjacent_delta_separate_keyframes_compact"), normalization_policy=POLICY,
                actions=actions, issues=issues, resolved_events=events)


def execute_boundary_plan(plan, signals, fps=30.0):
    result = resolve_plan(plan, signals, fps)
    if result["status"] != "resolved":
        raise ValueError(f"unresolved adjacent-delta boundary plan: {result['issues']}")
    return dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"], normalization_policy=POLICY)


def generate_boundary_plan(*, video: Path, bundle_file: Path, output: Path,
                           audit_dir: Path | None = None, max_repairs=3, call=None):
    if not 0 <= max_repairs <= 3:
        raise ValueError("max_repairs must be 0..3")
    audit = audit_dir or output.parent/"audit"
    audit.mkdir(parents=True, exist_ok=True)
    if output.exists() or list(audit.glob("attempt_*.response.txt")):
        raise FileExistsError("use fresh output/audit paths")
    legacy.load_env_file(Path(".env"))
    signals, _, fps = legacy.load_retargeting_bundle_signals(bundle_file)
    n = len(next(iter(signals.values())))
    available = set(signals)&legacy.DYNAMIC_SIGNALS
    base = build_prompt(available, n, fps)
    (audit/"base_prompt.txt").write_text(base)
    video_bytes = video.read_bytes()
    content = [{"type":"video_url", "video_url":{
        "url":"data:video/mp4;base64,"+base64.b64encode(video_bytes).decode("ascii")}}]
    metadata = dict(model=os.environ.get("OPENAI_MODEL"), prompt_version=PROMPT_VERSION,
                    source_video=str(video), source_video_sha256=hashlib.sha256(video_bytes).hexdigest(),
                    source_bundle=str(bundle_file), source_bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
                    source_frames=n, fps=fps, visual_source_kind="full_mp4_video", client_frame_sampling=False,
                    server_frame_sampling="unknown backend defaults", temperature=0, seed=0,
                    normalization_policy=POLICY, base_prompt_sha256=hashlib.sha256(base.encode()).hexdigest(),
                    numeric_parameters_per_boundary=1, numeric_parameters_per_action=2,
                    vlm_selects_time_or_reference=False, interior_predicate_required=False,
                    short_event_is_error=False)
    shared._write(audit/"metadata.json", metadata)
    previous, feedback = None, []
    request = call or legacy.call_vlm
    for attempt in range(max_repairs+1):
        prompt = base
        if attempt:
            prompt += ("\nPrevious candidate:\n"+json.dumps(previous, ensure_ascii=False)+
                       "\nALL local validation issues:\n"+json.dumps(feedback, ensure_ascii=False)+
                       "\nReview the full video and correct the separate start/end rules. "
                       "Each boundary has one signal, condition and threshold. Do not output "
                       "a time interval or a condition for an entire action. Do not delete "
                       "visible events merely to pass. A short/point event is permitted.")
        prefix = audit/f"attempt_{attempt}"
        prefix.with_suffix(".prompt.txt").write_text(prompt)
        raw = request(content, prompt)
        prefix.with_suffix(".response.txt").write_text(raw)
        try:
            previous = legacy._extract_json_value(raw)
            feedback = validation_issues(previous, available)
            if isinstance(previous, dict) and previous.get("schema") != SCHEMA:
                feedback.append(dict(field="schema", reason=f"new generation requires {SCHEMA}"))
            if feedback:
                shared._write(prefix.with_suffix(".validation.json"), feedback)
                continue
            shared._write(prefix.with_suffix(".plan.json"), previous)
            result = resolve_plan(previous, signals, fps)
            shared._write(prefix.with_suffix(".resolution.json"), result)
            feedback = result["issues"]
            if feedback:
                continue
            semantic = dict(fps=fps, events=result["resolved_events"], semantic_mode=result["semantic_mode"],
                            normalization_policy=POLICY, generation_metadata=dict(metadata, vlm_attempt_count=attempt+1,
                                                                                 visual_review="pending"))
            legacy.validate_semantic_keyframe_json(semantic, require_criticality=False)
            shared._write(output.with_name(output.stem+".event_plan.json"), previous)
            shared._write(output, semantic)
            return semantic
        except (ValueError, TypeError, KeyError) as exc:
            feedback = [dict(reason=str(exc))]
            shared._write(prefix.with_suffix(".validation.json"), feedback)
    raise ValueError(f"whole plan remains unresolved after {max_repairs+1} requests: {feedback}")
