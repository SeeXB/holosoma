"""One threshold per VLM action on normalized adjacent-frame differences.

Time, comparison reference and phase boundaries are not VLM outputs. The
executor matches maximal predicate runs in declared action order, preserving
the original GT sampling rate and every numeric decision for review.
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

SCHEMA = "holosoma.adjacent_delta_plan.v1"
PROMPT_VERSION = "adjacent_frame_single_threshold_v1"
POLICY = {
    "delta": "s[t] - s[t-1], t=1..N-1; frame zero has no predecessor",
    "ratio": "delta[t] / (max(s_whole_clip) - min(s_whole_clip))",
    "conditions": {"increasing": "d[t] >= threshold > 0", "decreasing": "d[t] <= -threshold < 0",
                   "steady": "abs(d[t]) <= threshold"},
    "matching": "first whole maximal contiguous true run whose first delta step is after previous action end",
    "window": "[first_matching_step-1, last_matching_step]; trigger=first_matching_step",
    "zero_span": "unresolved; no artificial denominator or fabricated ratio",
    "filters_or_dwell": "none",
    "fps_policy": "native GT rate, no resampling or multiplication by fps; thresholds are per frame",
    "unmatched": "unresolved; no nearest-value fallback, run trimming, or phase pruning",
}


def signal_guide(available):
    special = {
        "object_height": "object-origin Z displacement from first frame. Positive adjacent delta means rising; negative means lowering. Not ground clearance.",
        "object_displacement": "distance from initial object position. Its delta is change in that distance, not total motion or height.",
        "object_progress": "clipped displacement projection toward final object position, not elapsed time; its delta may increase or decrease.",
        "object_goal_distance": "distance to final object position. Negative delta means approaching that position, not necessarily placement/contact.",
        "object_rotate": "cumulative unsigned rotation. Nonnegative delta measures rotation amount in this step, not rotation direction.",
        "body_travel_progress": "normalized cumulative mean travel of all joints. Nonnegative delta measures movement in this step, not elapsed time.",
        "min_hand_object_distance": "minimum of the two hand-to-object-origin distances at each frame. Delta measures approach/separation, not surface contact.",
        "max_hand_object_distance": "maximum of the two hand-to-object-origin distances at each frame. Delta measures approach/separation, not surface contact.",
    }
    lines = []
    for name in sorted(available):
        if name in special:
            desc = special[name]
        elif name.endswith("_speed"):
            desc = ("UNSIGNED speed. Positive delta means speeding up, negative means slowing down. "
                    "Near-zero delta means speed nearly constant, whether moving or stopped; it does NOT imply rest.")
            if name == "object_vertical_speed":
                desc += " Both upward and downward motion have positive raw speed; its delta does not give vertical direction."
        elif name.endswith("_height"):
            desc = "world Z of the named joint. Positive delta means that region rises, negative means it lowers."
        elif name.endswith("_object_distance"):
            desc = "named joint-to-object-ORIGIN distance. Negative delta means approaching, positive means separating; not mesh contact or force."
        else:
            raise ValueError(f"missing signal description: {name}")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def build_prompt(available: set[str], frame_count: int, fps: float) -> str:
    return f"""Watch the entire supplied human-object video. GT has {frame_count} frames at {fps:g} fps.
Return visible semantic actions in chronological order, important body regions,
and ONE scalar threshold for EACH action. Do not assume a fixed lift/carry/place
sequence. Preserve distinct visible phases and repetitions; do not invent them.

The comparison is FIXED to consecutive GT frames, never selected by you:
  D = max(s over the WHOLE clip) - min(s over the WHOLE clip)
  d[t] = (s[t] - s[t-1]) / D, for t=1,...,N-1
There is no delta for frame 0. D is the raw signal's full-clip amplitude, NOT the
range of its differences. No division by the previous value, no cumulative
difference from the first frame, and no chosen reference/window is used.
You must NOT output frames, timestamps, time intervals, references, lag lengths,
start/end rules, clip_start/clip_end, or separate onset/exit thresholds.

For each action select ONE signal and ONE condition, plus ONE threshold number:
- increasing: d[t] >= threshold, threshold > 0.
- decreasing: d[t] <= -threshold, threshold > 0. The threshold is a POSITIVE
  magnitude; the condition supplies the negative sign.
- steady: abs(d[t]) <= threshold, threshold >= 0.
All thresholds are dimensionless numbers in [0,1]. They describe PER-FRAME
change, not the total change over an action, a percentile, or elapsed time.
Example of arithmetic only: if a signal covers its full amplitude uniformly in
100 adjacent steps, each d is 0.01, not 1. A threshold of 0.02 requires a 2% full
amplitude change in ONE step. Do not reuse large cumulative-phase percentages.
Use ONE threshold field, not separate positive/negative or start/end values.

EXECUTION: the program finds maximal consecutive runs satisfying each condition.
It selects the first complete run after the preceding action ends, in your list
order. A run on steps a..b covers GT poses a-1..b. Start, end and trigger frames
are computed locally. A run starting before the previous action ends is not cut
to invent another event. No minimum run length, smoothing, noise threshold,
hysteresis, search window or fallback is added. A one-step run remains visible
in the audit and is not proof of a meaningful phase. If no run matches, the action
is unresolved. Do not delete a visible action or substitute a vague whole-clip
action merely to pass. If D=0, normalized change is undefined and cannot resolve.

MEANING: height delta gives up/down; distance delta gives approach/separation.
Speed delta describes SPEED CHANGE, not movement direction. Constant speed is
not the same as standing still. A steady object height is compatible with either
holding, horizontal carrying, or an object resting on the ground; the video and
action order determine the semantic label, not this scalar alone. Near-flat GT
signals can normalize tiny jitter into large ratios. Do not invent a lift from
a visually flat height signal. Origin distance is not surface contact or force.

Return only this JSON object; no prose, Python, Markdown, or extra numeric fields:
{{"schema":"{SCHEMA}","actions":[{{
  "action":"<short action name>",
  "body_parts":["<allowed body region>"],
  "signal":"<available signal>",
  "condition":"<increasing, decreasing, or steady>",
  "threshold":"<one JSON number>",
  "rationale":"<visible action/anatomy evidence and why this signal/condition fits>"
}}]}}
The threshold placeholder must be replaced by a NUMBER, not a string or percent
text. There is deliberately no criticality number to estimate in this experiment;
the exporter uses a uniform placeholder weight and does not install training plans.
Allowed body_parts: {', '.join(sorted(legacy.MAPPABLE_BODY_PARTS))}.
The object is not a body part. Hand=palm/fingers; wrist=wrist region; elbow includes
forearm support. Select regions from visible evidence, not object-specific hints.

AVAILABLE SIGNALS (raw meanings and meanings of their adjacent-frame deltas):
{signal_guide(available)}
"""


def validation_issues(plan, available):
    if not isinstance(plan, dict) or set(plan) != {"schema", "actions"} or plan.get("schema") != SCHEMA:
        return [{"field":"plan", "reason":f"expected only schema={SCHEMA} and actions"}]
    if not isinstance(plan["actions"], list) or not plan["actions"]:
        return [{"field":"actions", "reason":"nonempty list required"}]
    issues = []
    for i, action in enumerate(plan["actions"]):
        field = f"actions[{i}]"
        if not isinstance(action, dict) or set(action) != {
                "action", "body_parts", "signal", "condition", "threshold", "rationale"}:
            issues.append(dict(field=field, reason="only action/body_parts/signal/condition/threshold/rationale; no time, reference, start/end or other quantities"))
            continue
        for key in ("action", "rationale"):
            if not isinstance(action[key], str) or not action[key].strip():
                issues.append(dict(field=field+"."+key, reason="nonempty string required"))
        parts = action["body_parts"]
        if (not isinstance(parts, list) or not parts or any(not isinstance(x,str) for x in parts)
                or len(parts)!=len(set(parts)) or set(parts)-legacy.MAPPABLE_BODY_PARTS):
            issues.append(dict(field=field+".body_parts", reason="use unique allowed body regions only; object is not a body region"))
        signal = action["signal"]
        if not isinstance(signal,str) or signal not in available:
            issues.append(dict(field=field+".signal", reason=f"unavailable signal: {signal}"))
        condition = action["condition"]
        if condition not in ("increasing", "decreasing", "steady"):
            issues.append(dict(field=field+".condition", reason="must be increasing, decreasing or steady"))
        threshold = action["threshold"]
        if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            issues.append(dict(field=field+".threshold", reason="one finite JSON number in [0,1] required"))
        elif condition != "steady" and threshold == 0:
            issues.append(dict(field=field+".threshold", reason="increasing/decreasing need a strictly positive magnitude"))
        if signal in ("object_rotate", "body_travel_progress") and condition == "decreasing":
            issues.append(dict(field=field, reason="cumulative nondecreasing signals cannot decrease; their increments can vary"))
    return issues


def delta_ratios(values):
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 1 or len(raw)<2 or not np.isfinite(raw).all():
        raise ValueError("finite 1D signal with at least two samples required")
    span = float(np.max(raw)-np.min(raw))
    if not math.isfinite(span):
        raise ValueError("signal range overflow")
    # Element zero is deliberately NaN: never manufacture a preceding pose.
    delta = np.r_[np.nan, np.diff(raw)]
    return (None if span==0 else delta/span), span


def true_runs(mask):
    """Inclusive maximal runs in delta-step coordinates, with step zero excluded."""
    mask = np.asarray(mask, dtype=bool).copy()
    mask[0] = False
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [(int(a),int(b-1)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1))]


def resolve_action(action, signals, minimum_step=1):
    raw = np.asarray(signals[action["signal"]], dtype=float)
    ratios, span = delta_ratios(raw)
    trace = dict(action=action["action"], signal=action["signal"], condition=action["condition"],
                 threshold=action["threshold"], status="unresolved", window=None, trigger_frame=None,
                 minimum_step=int(minimum_step), signal_min=float(np.min(raw)), signal_max=float(np.max(raw)),
                 signal_span=span, raw_step_threshold=float(action["threshold"]*span),
                 candidate_runs=[], eligible_runs=[], warnings=[])
    if ratios is None:
        trace["reason"] = "constant signal: normalized adjacent difference is undefined"
        return trace
    threshold = action["threshold"]
    if action["condition"] == "increasing":
        mask = ratios >= threshold
    elif action["condition"] == "decreasing":
        mask = ratios <= -threshold
    else:
        mask = np.abs(ratios) <= threshold
    runs = true_runs(mask)
    eligible = [(a,b) for a,b in runs if a>=minimum_step]
    trace.update(observed_delta_ratio_range=[float(np.min(ratios[1:])),float(np.max(ratios[1:]))],
                 candidate_runs=[[a,b] for a,b in runs], eligible_runs=[[a,b] for a,b in eligible],
                 whole_clip_satisfying_steps=int(np.count_nonzero(mask)), source_steps=len(raw)-1)
    if not eligible:
        trace["reason"] = "no complete matching run after previous action" if runs else "no adjacent-frame differences satisfy threshold"
        return trace
    a,b = eligible[0]
    # The window includes the pose before the first measured difference.
    trace.update(status="gt_resolved", window=[a-1,b], trigger_frame=a, matching_steps=b-a+1,
                 selected_run=[a,b], multiple_candidates=len(eligible)>1,
                 left_clip_censored=a==1, right_clip_censored=b==len(raw)-1,
                 first_ratio=float(ratios[a]), last_ratio=float(ratios[b]),
                 matched_ratio_range=[float(np.min(ratios[a:b+1])),float(np.max(ratios[a:b+1]))])
    if a==b:
        trace["warnings"].append("single adjacent step; not evidence of a sustained semantic phase")
    if a==1 and b==len(raw)-1:
        trace["warnings"].append("condition holds throughout video; no internal segmentation boundary")
    if len(eligible)>1:
        trace["warnings"].append("multiple candidate runs; earliest chosen by declared action order only")
    return trace


def resolve_plan(plan, signals, fps):
    if not signals or not math.isfinite(fps) or fps<=0:
        raise ValueError("nonempty signals and finite positive fps required")
    n = len(next(iter(signals.values())))
    if n<2 or any(np.asarray(v).shape!=(n,) or not np.isfinite(v).all() for v in signals.values()):
        raise ValueError("finite aligned 1D signals required")
    issues = validation_issues(plan,set(signals)&legacy.DYNAMIC_SIGNALS)
    if issues:
        raise ValueError(json.dumps(issues,ensure_ascii=False))
    actions, events, failures = [], [], []
    cursor = 1
    blocked = False
    for i, action in enumerate(plan["actions"]):
        trace = resolve_action(action,signals,cursor)
        trace["action_index"] = i
        if blocked:
            trace["diagnostic_candidate_run"] = trace.pop("selected_run",None)
            trace["status"] = "blocked_by_previous_action"
            trace["reason"] = "earlier action unresolved; candidate runs retained for diagnosis only"
            trace["window"] = None
            trace["trigger_frame"] = None
        actions.append(trace)
        if trace["status"] != "gt_resolved":
            blocked = True
            failures.append(dict(action_index=i,action=action["action"],reason=trace["reason"]))
            continue
        start,end = trace["window"]
        cursor = end+1
        start_evidence = dict(status="clip_censored" if trace["left_clip_censored"] else "gt_resolved",
                              frame=start, measured_delta_step=trace["trigger_frame"], ratio=trace["first_ratio"])
        end_evidence = dict(status="clip_censored" if trace["right_clip_censored"] else "gt_resolved",
                            frame=end, ratio=trace["last_ratio"])
        events.append(dict(event=f"{i:02d}_{action['action']}",action=action["action"],body_parts=action["body_parts"],
                           # Compatibility-only uniform weight, not another VLM number.
                           criticality_level=3,criticality=legacy.CRITICALITY_MAPPING[3],
                           criticality_source="uniform experiment placeholder; not VLM selected",
                           rationale=action["rationale"],predicate_evidence=trace,
                           boundary_evidence=dict(start=start_evidence,end=end_evidence),
                           windows=[dict(start_frame=start,end_frame=end,trigger_frame=trace["trigger_frame"])]))
    return dict(schema=SCHEMA,status="resolved" if not failures else "unresolved",fps=fps,source_frames=n,
                semantic_mode="vlm_adjacent_frame_single_threshold",normalization_policy=POLICY,
                actions=actions,issues=failures,resolved_events=events)


def execute_adjacent_delta_plan(plan, signals, fps=30.0):
    result = resolve_plan(plan,signals,fps)
    if result["status"] != "resolved":
        raise ValueError(f"unresolved adjacent-delta plan: {result['issues']}")
    return dict(fps=fps,events=result["resolved_events"],semantic_mode=result["semantic_mode"],normalization_policy=POLICY)


def _write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def generate_adjacent_delta_plan(*, video: Path, bundle_file: Path, output: Path,
                                 audit_dir: Path | None=None, max_repairs=3, call=None):
    if not 0<=max_repairs<=3:
        raise ValueError("max_repairs must be 0..3")
    audit = audit_dir or output.parent/"audit"
    audit.mkdir(parents=True,exist_ok=True)
    if output.exists() or list(audit.glob("attempt_*.response.txt")):
        raise FileExistsError("use fresh output/audit paths")
    legacy.load_env_file(Path(".env"))
    signals,_,fps = legacy.load_retargeting_bundle_signals(bundle_file)
    n = len(next(iter(signals.values())))
    available = set(signals)&legacy.DYNAMIC_SIGNALS
    base = build_prompt(available,n,fps)
    (audit/"base_prompt.txt").write_text(base)
    video_bytes = video.read_bytes()
    content = [{"type":"video_url","video_url":{
        "url":"data:video/mp4;base64,"+base64.b64encode(video_bytes).decode("ascii")}}]
    metadata = dict(model=os.environ.get("OPENAI_MODEL"),prompt_version=PROMPT_VERSION,
                    source_video=str(video),source_video_sha256=hashlib.sha256(video_bytes).hexdigest(),
                    source_bundle=str(bundle_file),source_bundle_sha256=hashlib.sha256(bundle_file.read_bytes()).hexdigest(),
                    source_frames=n,fps=fps,visual_source_kind="full_mp4_video",client_frame_sampling=False,
                    server_frame_sampling="unknown backend defaults",temperature=0,seed=0,
                    normalization_policy=POLICY,base_prompt_sha256=hashlib.sha256(base.encode()).hexdigest(),
                    numeric_parameters_per_action=1,vlm_selects_time_or_reference=False)
    _write(audit/"metadata.json",metadata)
    previous,feedback = None,[]
    request = call or legacy.call_vlm
    for attempt in range(max_repairs+1):
        prompt = base
        if attempt:
            prompt += ("\nPrevious candidate:\n"+json.dumps(previous,ensure_ascii=False)+
                       "\nALL local validation issues:\n"+json.dumps(feedback,ensure_ascii=False)+
                       "\nReview the full video and correct signal/condition/threshold choices. "
                       "Return the complete plan, exactly ONE numeric threshold per action. "
                       "Do not add timestamps/references or delete visible phases merely to pass. "
                       "Remember these are single-step changes, not cumulative phase changes.")
        prefix = audit/f"attempt_{attempt}"
        prefix.with_suffix(".prompt.txt").write_text(prompt)
        raw = request(content,prompt)
        prefix.with_suffix(".response.txt").write_text(raw)
        try:
            previous = legacy._extract_json_value(raw)
            feedback = validation_issues(previous,available)
            if feedback:
                _write(prefix.with_suffix(".validation.json"),feedback)
                continue
            _write(prefix.with_suffix(".plan.json"),previous)
            result = resolve_plan(previous,signals,fps)
            _write(prefix.with_suffix(".resolution.json"),result)
            feedback = result["issues"]
            if feedback:
                continue
            semantic = dict(fps=fps,events=result["resolved_events"],semantic_mode=result["semantic_mode"],
                            normalization_policy=POLICY,generation_metadata=dict(metadata,vlm_attempt_count=attempt+1,
                                                                                 visual_review="pending"))
            legacy.validate_semantic_keyframe_json(semantic)
            _write(output.with_name(output.stem+".event_plan.json"),previous)
            _write(output,semantic)
            return semantic
        except (ValueError,TypeError,KeyError) as exc:
            feedback = [dict(reason=str(exc))]
            _write(prefix.with_suffix(".validation.json"),feedback)
    raise ValueError(f"whole plan remains unresolved after {max_repairs+1} requests: {feedback}")
