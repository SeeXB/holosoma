"""Whole-clip VLM event programs evaluated only on full-rate trajectory signals.

No visual frame anchors, phase partitioning, quantile fallback, or coverage target.
Contact signals are anatomical point/segment proxies, not measured contact forces.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import re

import numpy as np

from .body_constraints import apply_body_part_constraints
from .pipeline import CRITICALITY_MAPPING, MAPPABLE_BODY_PARTS, load_retargeting_bundle_signals, _extract_json_value

SCHEMA = 'holosoma.trajectory_event_program.v1'


def unpack_event_response(raw):
    """One program contract for generation and SR; unwrap legacy transport only.

    No event fields or predicate values are repaired. The path points into the
    actual response so diagnostics identify the original location.
    """
    parsed = _extract_json_value(raw)
    if isinstance(parsed, dict) and set(parsed) == {'critique', 'program'}:
        return parsed['program'], parsed['critique'], '$.program'
    return parsed, None, '$'


def load_event_signals(bundle_file):
    """Use real triangle surfaces; legacy object-origin distance aliases are omitted."""
    signals, _, fps = load_retargeting_bundle_signals(bundle_file)
    signals = {k: v for k, v in signals.items()
               if 'distance' not in k and 'progress' not in k}
    units = {k: 'm/s' if 'speed' in k else 'm' for k in signals}
    if 'object_angular_speed' in units:
        units['object_angular_speed'] = 'rad/s'
    if 'object_rotate' in units:
        units['object_rotate'] = ('rad; cumulative unsigned 3-D orientation change from clip start; '
                                  'monotonic and quaternion-sign invariant')
    with np.load(bundle_file, allow_pickle=False) as data:
        if 'object_vertices' not in data:
            return signals, units, fps
        import igl
        joints = np.asarray(data['human_joints'], dtype=float)
        names = list(data['smplh_joint_names'])
        vertices = np.asarray(data['object_vertices'], dtype=float)
        faces = np.asarray(data['object_faces'], dtype=np.int32)
        local_vertices = np.asarray(data['object_vertices_local'], dtype=float)
        rotation = np.asarray(data['object_rotation'], dtype=float)
        translation = np.asarray(data['object_translation'], dtype=float)
        scale = np.asarray(data['object_scale'], dtype=float)
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError('Object scales must be positive')
        check = np.array([0, len(vertices)//2, len(vertices)-1])
        reconstructed = scale[check, None, None]*np.einsum('tij,vj->tvi', rotation[check], local_vertices)+translation[check, None]
        if not np.allclose(reconstructed, vertices[check], atol=2e-5):
            raise ValueError('Object surface transform does not match stored world vertices')
        obj = np.asarray(data['object_poses_wxyz_xyz'], dtype=float)[:, 4:7]
        signals['object_vertical_velocity'] = np.gradient(obj[:, 2]) * fps
        units['object_vertical_velocity'] = 'm/s (signed; upward positive)'
        # Sample the wrist-to-middle-finger segment and elbow-to-wrist segment.
        # This avoids interpreting distance to an object's origin as contact.
        for side, short in [('left', 'L'), ('right', 'R')]:
            wrist = joints[:, names.index(short+'_Wrist')]
            elbow = joints[:, names.index(short+'_Elbow')]
            finger = joints[:, names.index(short+'_Middle3')]
            proxies = {
                'hand': wrist[:, None] + np.linspace(0, 1, 9)[None, :, None]*(finger-wrist)[:, None],
                'elbow': elbow[:, None] + np.linspace(0, 1, 17)[None, :, None]*(wrist-elbow)[:, None],
                'wrist': wrist[:, None],
            }
            for part, points in proxies.items():
                local = np.einsum('tji,tkj->tki', rotation, (points-translation[:, None])/scale[:, None, None])
                squared, _, _ = igl.point_mesh_squared_distance(local.reshape(-1, 3), local_vertices, faces)
                values = np.sqrt(np.maximum(squared.reshape(len(joints), -1), 0)).min(axis=1)*scale
                key = f'{side}_{part}_surface_distance'
                signals[key] = np.asarray(values)
                units[key] = 'm; unsigned anatomical proxy to actual object triangles'
    return signals, units, fps


def predicate(expr, signals, *, contact_only=False, depth=0):
    """A restricted boolean expression; no eval, frame numbers, or arbitrary code."""
    if not isinstance(expr, dict) or depth > 8:
        raise ValueError('Invalid or excessively nested predicate')
    if set(expr) in ({'all'}, {'any'}):
        operator = next(iter(expr))
        children = expr[operator]
        if not isinstance(children, list) or not 1 <= len(children) <= 12:
            raise ValueError('all/any requires 1-12 predicates')
        values = [predicate(child, signals, contact_only=contact_only, depth=depth+1) for child in children]
        return np.logical_and.reduce(values) if operator == 'all' else np.logical_or.reduce(values)
    if set(expr) != {'signal', 'op', 'value'} or expr['signal'] not in signals:
        raise ValueError(f'Unknown predicate keys/signal: {expr}')
    value = expr['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Threshold must be a finite physical value')
    operations = {'gt': np.greater, 'ge': np.greater_equal, 'lt': np.less, 'le': np.less_equal}
    if expr['op'] not in operations:
        raise ValueError('Comparison op must be gt/ge/lt/le')
    if contact_only and (not expr['signal'].endswith('_surface_distance') or
                         expr['op'] not in ('lt', 'le') or not 0 < value <= 0.08):
        raise ValueError(f'Invalid contact predicate {expr}: unsigned distance is never <0; use lt/le with strictly positive threshold <=0.08 m, never center distance; contact requires surface proximity')
    return operations[expr['op']](signals[expr['signal']], value)


def seconds_frames(value, fps, name, *, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite seconds')
    if not (0 <= value <= 30) or (not allow_zero and value == 0):
        raise ValueError(f'{name} outside allowed range')
    return max(0 if allow_zero else 1, int(math.ceil(value*fps)))


def predicate_signals(expr):
    if 'signal' in expr:
        return {expr['signal']}
    return set().union(*(predicate_signals(child) for children in expr.values() for child in children))


def execute_event_program(plan, signals, fps):
    """Detect independent, possibly recurring intervals; keep unmatched events explicit."""
    if plan.get('schema') != SCHEMA or not isinstance(plan.get('actions'), list) or len(plan['actions']) > 16:
        raise ValueError('Expected a trajectory event program with at most 16 event functions')
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid fps')
    lengths = {len(v) for v in signals.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2 or any(not np.isfinite(v).all() for v in signals.values()):
        raise ValueError('Signals must be finite and aligned')
    n = next(iter(lengths))
    for action in plan['actions']:
        if not isinstance(action, dict) or set(action) != {
            'action', 'body_parts', 'criticality_level', 'rationale', 'function_rationale', 'recognition'
        }:
            raise ValueError('Event fields must describe evidence, body parts and recognition only; no visual timing')
    constrained = apply_body_part_constraints(plan)
    events, unresolved, diagnostics = [], [], []
    names = set()
    functions_seen = set()
    for action in constrained['actions']:
        name = action.get('action')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', name) or name in names:
            raise ValueError('Event names must be unique snake_case')
        names.add(name)
        if not action.get('body_parts') or set(action['body_parts']) - MAPPABLE_BODY_PARTS:
            raise ValueError(f'{name}: invalid body_parts {action.get("body_parts")}; '
                             f'choose only from {sorted(MAPPABLE_BODY_PARTS)}')
        level = action.get('criticality_level')
        if type(level) is not int or level not in CRITICALITY_MAPPING:
            raise ValueError('Invalid criticality level')
        for field in ('rationale', 'function_rationale'):
            if not isinstance(action.get(field), str) or not action[field].strip():
                raise ValueError(f'{name}: {field} is required')
        function = action['recognition']
        required = {'start', 'active', 'end', 'start_hold_s', 'end_hold_s', 'min_duration_s',
                    'max_duration_s', 'allow_initial', 'initial_evidence', 'contact',
                    'contact_timing', 'contact_lookback_s'}
        if not isinstance(function, dict) or set(function) != required:
            raise ValueError(f'{name}: recognition keys must be exactly {sorted(required)}')
        if type(function['allow_initial']) is not bool:
            raise ValueError('allow_initial must be boolean')
        if function['allow_initial'] and not function['initial_evidence']:
            raise ValueError('Initial event requires explicit video evidence')
        start = predicate(function['start'], signals)
        active = predicate(function['active'], signals)
        end = predicate(function['end'], signals)
        start_hold = seconds_frames(function['start_hold_s'], fps, 'start_hold_s')
        end_hold = seconds_frames(function['end_hold_s'], fps, 'end_hold_s')
        minimum = seconds_frames(function['min_duration_s'], fps, 'min_duration_s')
        maximum = seconds_frames(function['max_duration_s'], fps, 'max_duration_s')
        if maximum < minimum:
            raise ValueError('max_duration_s must be >= min_duration_s')
        contact_timing = function['contact_timing']
        contact = np.ones(n, dtype=bool)
        if contact_timing not in ('current', 'recent', 'none'):
            raise ValueError('contact_timing must be current/recent/none')
        if contact_timing != 'none':
            if not isinstance(function['contact'], dict):
                raise ValueError(f'{name}: recognition.contact cannot be null for {contact_timing}; provide a surface-distance predicate')
            contact = predicate(function['contact'], signals, contact_only=True)
            contact_names = predicate_signals(function['contact'])
            contact_parts = {s.removesuffix('_surface_distance') for s in contact_names}
            if contact_parts - set(action['body_parts']):
                raise ValueError(f'{name}: body_parts must include every contact region: {sorted(contact_parts)}')
            if plan.get('body_part_constraints', {}).get('mode') == 'bimanual_upper_limbs_v1':
                for region in ('hand', 'wrist', 'elbow'):
                    if (f'left_{region}_surface_distance' in contact_names) != (f'right_{region}_surface_distance' in contact_names):
                        raise ValueError(f'{name}: bimanual contact predicate must account for both {region} regions')
            if contact_timing == 'recent':
                lookback = seconds_frames(function['contact_lookback_s'], fps, 'contact_lookback_s')
                if function['contact_lookback_s'] > 0.5:
                    raise ValueError('Recent-contact history is limited to 0.5 seconds')
                contact = np.asarray([contact[max(0, i-lookback):i].any() for i in range(n)])
        elif function['contact'] is not None:
            raise ValueError('contact must be null for non-contact events')
        if 'object_height' in signals and contact_timing == 'none' and any(
            part.endswith(('_hand', '_elbow', '_wrist')) for part in action['body_parts']
        ):
            raise ValueError(f'{name}: object upper-limb event requires a surface contact gate')
        signature = json.dumps({k:function[k] for k in ('start','active','end','contact','contact_timing')}, sort_keys=True)
        if signature in functions_seen:
            raise ValueError(f'{name}: duplicate recognition functions cannot distinguish different events')
        functions_seen.add(signature)
        eligible = start & active & contact & ~end
        rising = eligible & ~np.r_[False, eligible[:-1]]
        if not function['allow_initial']:
            rising[0] = False
        windows, rejected, previous_end = [], [], -1
        for onset in np.flatnonzero(rising):
            onset = int(onset)
            if onset <= previous_end or onset+start_hold > n or not eligible[onset:onset+start_hold].all():
                continue
            stop, reason = n-1, 'clip_end'
            for t in range(onset+1, n):
                if not active[t] or (contact_timing == 'current' and not contact[t]):
                    stop, reason = t-1, 'active_or_contact_lost'
                    break
                if t+end_hold <= n and end[t:t+end_hold].all():
                    stop, reason = t-1, 'end_predicate'
                    break
            length = stop-onset+1
            if reason == 'clip_end':
                rejected.append(dict(start_frame=onset, end_frame=stop, reason='termination_not_observed'))
                continue
            if length < max(minimum, start_hold) or length > maximum:
                rejected.append(dict(start_frame=onset, end_frame=stop, reason='duration_out_of_bounds'))
                continue
            windows.append(dict(start_frame=onset, trigger_frame=onset, end_frame=stop))
            previous_end = stop
        diagnostics.append(dict(event=name, windows=windows, rejected=rejected))
        if not windows:
            unresolved.append(dict(event=name, reason='no sustained trajectory trigger/valid duration', rejected=rejected))
            continue
        events.append(dict(event=name, action=name, body_parts=action['body_parts'], confidence=1.0,
                           criticality_level=level, criticality=CRITICALITY_MAPPING[level],
                           criticality_rationale=action['function_rationale'],
                           rationale=action['rationale'], windows=windows,
                           recognition=deepcopy(function), boundary_source='trajectory_predicates'))
    covered = np.zeros(n, dtype=bool)
    for event in events:
        for window in event['windows']:
            covered[window['start_frame']:window['end_frame']+1] = True
    return dict(schema=SCHEMA, semantic_mode=SCHEMA, fps=fps, events=events,
                unresolved_events=unresolved, recognition_diagnostics=diagnostics,
                coverage=dict(frames=n, key_frames=int(covered.sum()), ordinary_frames=int((~covered).sum()),
                              fraction=float(covered.mean())))


def event_prompt(units, *, bimanual=False):
    """Whole-clip English instruction with a task-independent structural template."""
    object_task = 'object_height' in units
    atom = dict(signal='<name_from_available_signals>', op='<gt|ge|lt|le>', value='<finite_number>')
    template = dict(schema=SCHEMA, actions=[dict(
        action='<unique_snake_case_event_name>', body_parts=['<allowed_body_part>'],
        criticality_level='<integer_1_to_4>', rationale='<observed_visual_evidence>',
        function_rationale='<why_the_conditions_detect_this_event_and_its_end>',
        recognition=dict(start=atom, active='<predicate>', end='<predicate>',
                         start_hold_s='<positive_seconds>', end_hold_s='<positive_seconds>',
                         min_duration_s='<positive_seconds>', max_duration_s='<positive_seconds>',
                         allow_initial='<boolean>', initial_evidence='<string>',
                         contact='<surface_predicate_or_null>', contact_timing='<current|recent|none>',
                         contact_lookback_s='<nonnegative_seconds>'))])
    return (
        'Analyze ALL chronological images together as ONE whole video. Generate independently from '
        'these original inputs. Identify critical EVENTS, their body parts and trajectory recognition '
        'functions. Do not partition the timeline into phases. Preparation, waiting and routine motion '
        'may remain ordinary. A separate full-rate trajectory, not image timing, determines every interval.\n'
        'OUTPUT: one JSON OBJECT with exactly schema and actions, not a top-level array. Propose up to '
        'four distinct critical event types; one function can detect multiple occurrences. No markdown.\n'
        'STRUCTURAL TEMPLATE ONLY: angle-bracket placeholders specify types, not literal outputs. '
        'Replace every placeholder. Numbers and booleans must be JSON numbers and booleans, not strings. '
        'Each predicate placeholder expands to an atomic predicate or an all/any tree. '
        'There is no example task, contact region, threshold, duration or event annotation to copy.\n'
        + json.dumps(template, indent=2) + '\n'
        'FUNCTION RULES: start/active/end are predicates. A predicate is exactly '
        '{"signal":"<available_signal>","op":"<gt|ge|lt|le>","value":"<finite_number>"}, '
        'or {"all":["<predicate>","<predicate>"]} or {"any":["<predicate>","<predicate>"]}. '
        'Replace predicate placeholders with objects; all means AND, any means OR. '
        'Never combine all/any with other keys in the same object. Use ONLY signals listed below. '
        'Never invent angles, accelerations or velocity signals absent from the list. '
        'The anatomical label vocabulary and the signal vocabulary are different: a valid body part '
        'does not imply a signal exists for it. Positive object_vertical_velocity means upward; negative '
        'means downward. object_rotate is cumulative unsigned 3-D orientation change in radians from '
        'the clip start and never decreases; object_angular_speed is the instantaneous rotation rate. '
        'Motion-specific claims need motion conditions, not proximity alone. '
        'start triggers on a false-to-true transition with active/contact satisfied. '
        'active must remain true; end or loss of active/current contact terminates the event. '
        'start_hold_s, end_hold_s and min_duration_s must be strictly POSITIVE seconds. '
        'max_duration_s must be positive and >= min_duration_s, at most 30 seconds. '
        'These bounds validate a detected interval; they never create its endpoint. '
        'contact_lookback_s must be a NUMBER, never null; use 0 when unused. '
        'Keep allow_initial=false unless the first image proves this event is already active; '
        'then supply specific initial_evidence. No frame indices, time anchors, quantiles, '
        'extrema fallback or coverage target. Return actions=[] if no critical event is identifiable.\n'
        + ('OBJECT CONTACT: contact is a PREDICATE OBJECT, NEVER true or false. Grasp/lift/support/place '
           'require contact_timing="current" and surface proximity. Unsigned distance can never be <0: '
           'contact comparisons must use lt/le and a strictly positive threshold <=0.08 m. '
           'Release may use contact_timing="recent" with 0<contact_lookback_s<=0.5. '
           'Choose anatomical region from evidence: hand=palm/fingers, wrist=wrist, elbow=forearm '
           'segment; do not always choose the same region. Include contact regions in body_parts. '
           'Surface distances are point/segment proxies, not force measurements. '
           if object_task else
           'BODY-ONLY MOTION: there is no object. Use contact=null, contact_timing="none", '
           'contact_lookback_s=0. Choose body height/speed predicates for visible critical transitions. ')
        + ('USER-CONFIRMED BIMANUAL: pair every selected hand/wrist/elbow/shoulder with its other side. '
           'Contact predicates must account for BOTH sides of every selected contact region. '
           if bimanual else '')
        + '\nAvailable signals and units: ' + json.dumps(units)
        + '\nAllowed body_parts: ' + json.dumps(sorted(MAPPABLE_BODY_PARTS))
        + '\ncriticality_level must be an integer 1 to 4. All evidence and explanations must be in English.'
    )
