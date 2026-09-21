"""Field-localized SR feedback; inspect proposals without changing their semantics."""
from __future__ import annotations

import json
import math
import re

import numpy as np

from .pipeline import MAPPABLE_BODY_PARTS, CRITICALITY_MAPPING
from .trajectory_events import SCHEMA, execute_event_program, predicate, predicate_signals, seconds_frames, unpack_event_response

ACTION_KEYS = {'action', 'body_parts', 'criticality_level', 'rationale', 'function_rationale', 'recognition'}
FUNCTION_KEYS = {'start', 'active', 'end', 'start_hold_s', 'end_hold_s', 'min_duration_s',
                 'max_duration_s', 'allow_initial', 'initial_evidence', 'contact', 'contact_timing', 'contact_lookback_s'}


def safe_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k:safe_value(v) for k,v in value.items()}
    if isinstance(value, list):
        return [safe_value(v) for v in value]
    return value


def mask_summary(mask):
    mask = np.asarray(mask, dtype=bool)
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return dict(true_frames=int(mask.sum()), intervals=int(len(starts)),
                first_intervals=[dict(start_frame=int(a), end_frame=int(b)) for a,b in zip(starts[:6],ends[:6])])


def diagnose_response(raw, constraints, signals, fps, *, reflected=False):
    """Return all detected locations, not only the executor's first exception."""
    report = dict(errors=[], warnings=[], per_event=[], frame_indices='zero-based trajectory diagnostics, not visual anchors')

    def issue(path, actual, reason, event=None, *, warning=False, **detail):
        row = dict(path=path, event=event, actual=safe_value(actual), reason=reason, **detail)
        report['warnings' if warning else 'errors'].append(row)

    try:
        parsed, _, base = unpack_event_response(raw)
    except (ValueError, TypeError) as exc:
        detail = {}
        if isinstance(exc, json.JSONDecodeError):
            detail = dict(json_block_line=exc.lineno, json_block_column=exc.colno, json_block_offset=exc.pos)
        issue('$', raw[:300], str(exc), **detail)
        return report
    if not isinstance(parsed, dict):
        issue(base, parsed, 'Expected program object, not an array/string/null')
        return report
    for key in {'schema','actions'} - set(parsed):
        issue(base+'.'+key, None, 'Required field is missing')
    for key in set(parsed) - {'schema','actions'}:
        issue(base+'.'+key, parsed[key], 'Unexpected program field; intervals come from trajectory execution')
    if parsed.get('schema') != SCHEMA:
        issue(base+'.schema', parsed.get('schema'), 'Expected '+SCHEMA)
    actions = parsed.get('actions')
    if not isinstance(actions, list) or len(actions) > 16:
        issue(base+'.actions', actions, 'Expected a list of at most 16 critical event functions')
        return report
    names, signatures = {}, {}
    for index, action in enumerate(actions):
        path = f'{base}.actions[{index}]'
        event = action.get('action') if isinstance(action, dict) else None
        before = len(report['errors'])
        if not isinstance(action, dict):
            issue(path, action, 'Expected an event object', event)
            continue
        for key in ACTION_KEYS-set(action):
            issue(path+'.'+key, None, 'Required event field is missing', event)
        for key in set(action)-ACTION_KEYS:
            issue(path+'.'+key, action[key], 'Unexpected event field', event)
        if not isinstance(event, str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', event):
            issue(path+'.action', event, 'Expected unique snake_case event name', event)
        elif event in names:
            issue(path+'.action', event, 'Duplicate event name', event, conflicts_with=names[event])
        else:
            names[event] = path+'.action'
        if event == 'example_transition':
            issue(path+'.action', event, 'Syntax example must be replaced by a video-grounded event', event)
        parts = action.get('body_parts')
        if not isinstance(parts, list) or not parts:
            issue(path+'.body_parts', parts, 'Expected nonempty list of allowed anatomical parts', event)
        else:
            for k,part in enumerate(parts):
                if not isinstance(part, str) or part not in MAPPABLE_BODY_PARTS:
                    issue(f'{path}.body_parts[{k}]', part, 'Unknown anatomical label; select only listed names',
                          event, allowed=sorted(MAPPABLE_BODY_PARTS))
        level = action.get('criticality_level')
        if type(level) is not int or level not in CRITICALITY_MAPPING:
            issue(path+'.criticality_level', level, 'Expected integer 1 to 4', event)
        for key in ('rationale','function_rationale'):
            if not isinstance(action.get(key), str) or not action[key].strip():
                issue(path+'.'+key, action.get(key), 'Expected nonempty English evidence/explanation', event)
        function = action.get('recognition')
        if not isinstance(function, dict):
            issue(path+'.recognition', function, 'Expected recognition object', event)
            continue
        fp = path+'.recognition'
        for key in FUNCTION_KEYS-set(function):
            issue(fp+'.'+key, None, 'Required recognition field is missing', event)
        for key in set(function)-FUNCTION_KEYS:
            issue(fp+'.'+key, function[key], 'Unexpected recognition field', event)
        for key in ('start_hold_s','end_hold_s','min_duration_s','max_duration_s','contact_lookback_s'):
            if key not in function:continue
            try: seconds_frames(function[key],fps,key,allow_zero=key=='contact_lookback_s')
            except ValueError as exc: issue(fp+'.'+key, function[key], str(exc), event)
        low, high = function.get('min_duration_s'), function.get('max_duration_s')
        if type(low) in (float,int) and type(high) in (float,int) and high < low:
            issue(fp+'.max_duration_s', high, 'Maximum duration is below minimum duration', event, minimum=low)
        if type(function.get('allow_initial')) is not bool:
            issue(fp+'.allow_initial', function.get('allow_initial'), 'Expected boolean', event)
        if function.get('allow_initial') is True and not function.get('initial_evidence'):
            issue(fp+'.initial_evidence', function.get('initial_evidence'), 'Initial events require specific first-image evidence', event)
        timing = function.get('contact_timing')
        if timing not in ('current','recent','none'):
            issue(fp+'.contact_timing', timing, 'Expected current/recent/none', event)
        if timing=='none' and function.get('contact') is not None:
            issue(fp+'.contact', function.get('contact'), 'Non-contact events require contact=null', event)
        leaves = []

        def inspect_predicate(expr, location, contact=False, depth=0):
            if depth>8:
                issue(location, None, 'Predicate nesting exceeds 8', event);return
            if not isinstance(expr, dict):
                issue(location, expr, 'Expected predicate object, never boolean/null/array', event);return
            if set(expr) in ({'all'},{'any'}):
                operator = next(iter(expr)); children=expr[operator]
                if not isinstance(children,list) or not 1<=len(children)<=12:
                    issue(location+'.'+operator,children,'Expected 1 to 12 predicates',event);return
                for k,child in enumerate(children):inspect_predicate(child,f'{location}.{operator}[{k}]',contact,depth+1)
                return
            if set(expr)!={'signal','op','value'}:
                issue(location,expr,'Atomic predicate requires exactly signal/op/value',event);return
            try:
                matched=predicate(expr,signals,contact_only=contact)
                values=signals[expr['signal']]
                leaves.append(dict(path=location,signal=expr['signal'],threshold=expr['value'],
                                   signal_min=float(np.min(values)),signal_max=float(np.max(values)),**mask_summary(matched)))
            except (ValueError,TypeError,KeyError) as exc:
                suffix = ('.signal' if not isinstance(expr['signal'],str) or expr['signal'] not in signals
                          or (contact and not expr['signal'].endswith('_surface_distance')) else
                          '.op' if expr['op'] not in ('gt','ge','lt','le') or (contact and expr['op'] not in ('lt','le'))
                          else '.value')
                issue(location+suffix,expr.get(suffix[1:]),str(exc),event)

        for key in ('start','active','end'):
            if key in function: inspect_predicate(function[key],fp+'.'+key)
        if timing in ('current','recent'):
            inspect_predicate(function.get('contact'),fp+'.contact',True)
        if len(report['errors'])==before:
            contact_names=predicate_signals(function['contact']) if timing!='none' else set()
            contact_parts={name.removesuffix('_surface_distance') for name in contact_names}
            effective=set(parts)
            if constraints.get('mode')=='bimanual_upper_limbs_v1':
                for part in tuple(effective):
                    side,_,region=part.partition('_')
                    if side in ('left','right') and region in ('hand','wrist','elbow','shoulder'):
                        effective.add(('right' if side=='left' else 'left')+'_'+region)
                for region in ('hand','wrist','elbow'):
                    if (f'left_{region}_surface_distance' in contact_names)!=(f'right_{region}_surface_distance' in contact_names):
                        issue(fp+'.contact',function['contact'],f'Bimanual predicate must account for both {region} sides',event)
            if contact_parts-effective:
                issue(path+'.body_parts',parts,'Missing body parts selected by the contact predicate',event,missing=sorted(contact_parts-effective))
            if timing=='none' and 'object_height' in signals and any(p.endswith(('_hand','_wrist','_elbow')) for p in parts):
                issue(fp+'.contact_timing',timing,'Object upper-limb event requires a surface contact gate',event)
        if all(key in function for key in ('start','active','end','contact','contact_timing')):
            signature=json.dumps({k:function[k] for k in ('start','active','end','contact','contact_timing')},sort_keys=True)
            if signature in signatures:
                issue(fp,function,'Different events have identical detection functions; they cannot distinguish their claimed transitions',event,conflicts_with=signatures[signature])
            else:signatures[signature]=fp
        item=dict(path=path,event=event,predicate_checks=leaves)
        try:
            single=dict(schema=parsed.get('schema'),actions=[action])
            if constraints:single['body_part_constraints']=constraints
            result=execute_event_program(single,signals,fps)
            item.update(valid=True,coverage=result['coverage'],unresolved=result['unresolved_events'],
                        intervals=result['recognition_diagnostics'])
            masks={k:predicate(function[k],signals) for k in ('start','active','end')}
            contact=np.ones(len(next(iter(signals.values()))),dtype=bool)
            if timing!='none':
                contact=predicate(function['contact'],signals,contact_only=True)
                if timing=='recent':
                    span=seconds_frames(function['contact_lookback_s'],fps,'contact_lookback_s')
                    contact=np.asarray([contact[max(0,i-span):i].any() for i in range(len(contact))])
            masks['contact']=contact
            eligible=masks['start'] & masks['active'] & contact & ~masks['end']
            item['gate_counts']={key:int(mask.sum()) for key,mask in masks.items()}
            item['gate_counts']['joint_eligible']=int(eligible.sum())
            rising=eligible & ~np.r_[False,eligible[:-1]]
            if not function['allow_initial']:rising[0]=False
            item['candidate_trigger_frames']=np.flatnonzero(rising).tolist()[:12]
            hold=seconds_frames(function['start_hold_s'],fps,'start_hold_s')
            item['trigger_hold_checks']=[dict(frame=int(t), required_frames=hold,
                available_frames=min(hold,len(eligible)-int(t)),
                passed=bool(t+hold<=len(eligible) and eligible[t:t+hold].all()))
                for t in np.flatnonzero(rising)[:12]]
            if result['unresolved_events']:
                for key in ('start','active','contact'):
                    if not masks[key].any():
                        issue(fp+'.'+key,function.get(key),'This gate is false on every trajectory frame; check evidence/units, do not force a match',event,warning=True)
                if masks['end'].all():issue(fp+'.end',function['end'],'End is true on every frame, suppressing every trigger',event,warning=True)
                if not eligible.any():issue(fp,function,'Start, active, contact and not-end never hold simultaneously',event,warning=True)
                elif not rising.any():issue(fp+'.allow_initial',function['allow_initial'],'No eligible false-to-true transition after the initial frame',event,warning=True)
                else:issue(fp,function,'Candidate onsets exist but fail hold/duration/termination checks; inspect candidate_trigger_frames and rejected intervals',event,warning=True)
                for entry in result['recognition_diagnostics'][0]['rejected'][:6]:
                    if entry['reason']=='termination_not_observed':
                        issue(fp+'.end',function['end'],'No termination observed before trajectory ends; do not invent a clipped endpoint',event,warning=True,trajectory_interval=entry)
                    else:
                        length=entry['end_frame']-entry['start_frame']+1
                        bound='max_duration_s' if length>seconds_frames(function['max_duration_s'],fps,'max_duration_s') else 'min_duration_s'
                        issue(fp+'.'+bound,function[bound],'Detected interval violates duration bounds',event,warning=True,trajectory_interval=entry,detected_seconds=length/fps)
            used=set().union(*(predicate_signals(function[k]) for k in ('start','active','end')))
            claim=str(event)+' '+action['rationale']
            if used and all(name.endswith('_surface_distance') for name in used) and re.search(r'lift|rais|lower|carry|carrying|overhead',claim,re.I):
                issue(fp,function,'Motion-specific event is detected using proximity alone; the function cannot establish the claimed motion',event,warning=True)
            up=bool(re.search(r'lift|rais|pick.?up',claim,re.I));down=bool(re.search(r'lower|put.?down|place.?down',claim,re.I))
            start=function['start']
            if isinstance(start,dict) and start.get('signal')=='object_vertical_velocity':
                wrong=(up and not down and start['op'] in ('lt','le') and start['value']<0) or (down and not up and start['op'] in ('gt','ge') and start['value']>0)
                if wrong:issue(fp+'.start',start,'Signed vertical-velocity direction contradicts the event name/visual rationale (up positive, down negative)',event,warning=True)
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            item.update(valid=False,error=str(exc))
            if len(report['errors'])==before:issue(fp,function,str(exc),event)
        report['per_event'].append(item)
    return report
