"""Separate whole-video event discovery from trajectory predicate construction.

Both agents see the original visual inputs. Only the logic agent receives the
trajectory signal catalog; only the local executor determines intervals.
"""
from __future__ import annotations

import json
import re
from copy import deepcopy

from .pipeline import CRITICALITY_MAPPING, MAPPABLE_BODY_PARTS, _extract_json_value
from .trajectory_events import SCHEMA, event_prompt

EVENT_CATALOG_SCHEMA = 'holosoma.video_event_catalog.v1'
FUNCTION_SCHEMA = 'holosoma.event_recognition_functions.v1'
EVENT_FUNCTION_SCHEMA = FUNCTION_SCHEMA
EVENT_FIELDS = frozenset({'action', 'body_parts', 'criticality_level', 'rationale'})


def event_catalog_prompt(*, bimanual=False, object_task=True):
    """Task-independent structural contract; no trajectory thresholds or examples."""
    template = dict(schema=EVENT_CATALOG_SCHEMA, events=[dict(
        action='<unique_snake_case_event_name>', body_parts=['<allowed_body_part>'],
        criticality_level='<integer_1_to_4>', rationale='<observed_visual_evidence>')])
    return (
        'You are the VIDEO EVENT AGENT. Analyze ALL chronological images together as ONE whole '
        'video, independently from these original inputs. Identify one to four distinct critical '
        'event types and their key anatomical parts. One event type may recur. Do not divide the '
        'video into phases or attempt to cover the entire video. Ordinary preparation, waiting and '
        'routine motion may remain noncritical. Do not invent events to fill the catalog.\n'
        'Your responsibility ends at visual event discovery. Do not propose recognition functions, '
        'signal names, numerical thresholds, frame indices, timestamps, durations or key intervals. '
        'A separate logic agent will construct functions and a local trajectory executor will '
        'determine intervals. Rationale must describe visible evidence, not an assumed timeline.\n'
        'OUTPUT: one JSON object with exactly schema and events. Each event has exactly action, '
        'body_parts, criticality_level and rationale. Event names must be unique snake_case; '
        'body_parts must be a nonempty list from the vocabulary below; criticality_level is an '
        'integer from 1 to 4; rationale is nonempty English visual evidence. No markdown.\n'
        'STRUCTURAL TEMPLATE ONLY: replace every angle-bracket placeholder. These are type '
        'descriptions, not literal values or a concrete event example. Numbers must be JSON numbers.\n'
        + json.dumps(template, indent=2)
        + ('\nOBJECT INTERACTION: key parts must include visible object-contact regions. '
           'hand denotes palm/fingers, wrist denotes the wrist region, elbow denotes the forearm '
           'segment. Select contact regions from the visual evidence; do not substitute wrist or '
           'elbow for visible hand contact. Include other parts only when key to the event. Treat a '
           'clearly visible, critical change in object orientation as distinct from translation or '
           'lifting instead of hiding it inside a generic move event.'
           if object_task else '\nBODY-ONLY MOTION: select the parts central to the visible event.')
        + ('\nUSER-CONFIRMED BIMANUAL: every selected hand, wrist, elbow or shoulder must include '
           'both its left and right labels. This is a task constraint, not a requirement to infer '
           'image left/right. Include both explicitly; the validator will not add missing labels.'
           if bimanual else '')
        + '\nAllowed body_parts: ' + json.dumps(sorted(MAPPABLE_BODY_PARTS))
    )


def diagnose_event_catalog(raw, constraints):
    """Return the unchanged catalog only when valid, plus field-localized feedback."""
    report = dict(errors=[], warnings=[], per_event=[])

    def issue(path, actual, reason, **details):
        report['errors'].append(dict(path=path, actual=actual, reason=reason, **details))

    try:
        parsed = _extract_json_value(raw)
    except (ValueError, TypeError) as exc:
        details = {}
        if isinstance(exc, json.JSONDecodeError):
            details = dict(json_block_line=exc.lineno, json_block_column=exc.colno,
                           json_block_offset=exc.pos)
        issue('$', str(raw)[:300], str(exc), **details)
        return None, report
    if not isinstance(parsed, dict):
        issue('$', parsed, 'Expected one event catalog object')
        return None, report
    for field in sorted({'schema', 'events'} - set(parsed)):
        issue('$.' + field, None, 'Required catalog field is missing')
    for field in sorted(set(parsed) - {'schema', 'events'}):
        issue('$.' + field, parsed[field], 'Unexpected catalog field; this agent supplies no timing or functions')
    if parsed.get('schema') != EVENT_CATALOG_SCHEMA:
        issue('$.schema', parsed.get('schema'), 'Expected ' + EVENT_CATALOG_SCHEMA)
    events = parsed.get('events')
    if not isinstance(events, list) or not 1 <= len(events) <= 4:
        issue('$.events', events, 'Expected one to four visually supported critical event types')
        return None, report
    mode = (constraints or {}).get('mode')
    if mode not in (None, 'bimanual_upper_limbs_v1'):
        issue('$.events', mode, 'Unsupported external body part constraint')
    names = {}
    for index, event in enumerate(events):
        path = f'$.events[{index}]'
        before = len(report['errors'])
        if not isinstance(event, dict):
            issue(path, event, 'Expected an event object')
            continue
        for field in sorted(EVENT_FIELDS - set(event)):
            issue(path + '.' + field, None, 'Required event field is missing')
        for field in sorted(set(event) - EVENT_FIELDS):
            issue(path + '.' + field, event[field], 'Unexpected event field; timing and recognition belong to later stages')
        name = event.get('action')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]{1,63}', name):
            issue(path + '.action', name, 'Expected unique snake_case event name')
        elif name in names:
            issue(path + '.action', name, 'Duplicate event name', conflicts_with=names[name])
        else:
            names[name] = path + '.action'
        parts = event.get('body_parts')
        if not isinstance(parts, list) or not parts:
            issue(path + '.body_parts', parts, 'Expected a nonempty list of allowed anatomical labels')
        else:
            valid_parts = set()
            for part_index, part in enumerate(parts):
                part_path = f'{path}.body_parts[{part_index}]'
                if not isinstance(part, str) or part not in MAPPABLE_BODY_PARTS:
                    issue(part_path, part, 'Unknown anatomical label', allowed=sorted(MAPPABLE_BODY_PARTS))
                    continue
                if part in valid_parts:
                    issue(part_path, part, 'Duplicate anatomical label')
                valid_parts.add(part)
            if mode == 'bimanual_upper_limbs_v1':
                for part in sorted(valid_parts):
                    side, _, region = part.partition('_')
                    if side in ('left', 'right') and region in ('hand', 'wrist', 'elbow', 'shoulder'):
                        partner = ('right' if side == 'left' else 'left') + '_' + region
                        if partner not in valid_parts:
                            issue(path + '.body_parts', parts, 'User-confirmed bimanual event must include both sides of each selected upper-limb region', missing=partner)
        if type(event.get('criticality_level')) is not int or event['criticality_level'] not in CRITICALITY_MAPPING:
            issue(path + '.criticality_level', event.get('criticality_level'), 'Expected integer 1 to 4')
        if not isinstance(event.get('rationale'), str) or not event['rationale'].strip():
            issue(path + '.rationale', event.get('rationale'), 'Expected nonempty English visual evidence')
        report['per_event'].append(dict(path=path, event=name, errors=len(report['errors']) - before))
    return (None if report['errors'] else parsed), report


def logic_agent_prompt(catalog, units, ranges, *, bimanual=False):
    """The second agent returns functions only; catalog metadata remains immutable."""
    # Reuse exactly the executor's recognition rules, without its single-agent
    # discovery instructions or program template.
    rules = event_prompt(units, bimanual=bimanual).split('FUNCTION RULES: ', 1)[1]
    rules = rules.split('\nAllowed body_parts:', 1)[0]
    rules = rules.replace(
        'Return actions=[] if no critical event is identifiable.',
        'Never omit a catalog event when its function is difficult to construct.')
    template = """def EVENT_NAME(signals):
    return {
        "function_rationale": "<why these conditions identify this event>",
        "recognition": {
            "start": signals["<available_signal>"] > NUMBER,
            "active": (signals["<available_signal>"] > NUMBER) & (signals["<available_signal>"] < NUMBER),
            "end": signals["<available_signal>"] <= NUMBER,
            "contact": (signals["<contact_signal>"] <= NUMBER) | (signals["<contact_signal>"] <= NUMBER),
            "start_hold_s": SECONDS,
            "end_hold_s": SECONDS,
            "min_duration_s": SECONDS,
            "max_duration_s": SECONDS,
            "allow_initial": False,
            "initial_evidence": "<evidence if initially active, otherwise empty>",
            "contact_timing": "<current|recent|none>",
            "contact_lookback_s": SECONDS,
        },
    }
"""
    # Retain physical rules, but remove the obsolete JSON-predicate grammar.
    rules = rules[rules.index('Never invent angles,'):]
    rules = rules.replace('contact is a PREDICATE OBJECT, NEVER true or false.',
                          'contact is a Python boolean array expression, NEVER True or False.')
    rules = rules.replace('contact=null', 'contact=None').replace('allow_initial=false', 'allow_initial=False')
    rules = rules.replace('contact comparisons must use lt/le', 'contact comparisons must use < or <=')
    return (
        'You are the TRAJECTORY LOGIC AGENT. Write PYTHON SOURCE CODE, not JSON and not pseudocode. '
        'Analyze all attached chronological images as one whole video and use the fixed event catalog. '
        'Return one Python function def EXACT_EVENT_NAME(signals): for EACH catalog event. '
        'The event names, body parts and evidence are fixed; do not add, remove or reinterpret events. '
        'Return only Python code. No JSON schema/functions wrapper or Markdown commentary.\n'
        'Each function has one return dictionary with function_rationale and recognition. '
        'Use Python True, False and None. Conditions in start/active/end/contact must be Python '
        'expressions over NumPy signal arrays: signals["available_name"] compared with a finite '
        'literal number using >, >=, < or <=. Combine parenthesized comparisons using & (AND) '
        'and | (OR). Do not use Python and/or with arrays, chained comparisons, JSON all/any trees, '
        'imports, function calls, assignments, loops, frame indexing or arbitrary code. '
        'Use literals for timing fields. No frame anchors or fabricated windows. '
        'Explain the event-specific conditions in function_rationale in English.\n'
        'STRUCTURAL TEMPLATE ONLY: EVENT_NAME, NUMBER, SECONDS and angle-bracket labels are '
        'placeholders. Replace them with the exact event name, finite numeric literals and available '
        'signal names. NUMBER has type <finite_number>; SECONDS has type <positive_seconds> '
        '(contact_lookback_s may be zero). No particular event or threshold is supplied. '
        'The operators in the template illustrate syntax, not the correct conditions for your event.\n'
        + template + '\nFIXED EVENT CATALOG:\n' + json.dumps(catalog, indent=2)
        + '\nPHYSICAL AND INTERVAL RULES: ' + rules
        + '\nObserved signal ranges (physical values, not event labels): ' + json.dumps(ranges)
    )


def syntax_repair_agent_prompt(source, errors, *, round_index, budget):
    """Build a text-only, semantics-preserving repair request for Python source."""
    return (
        'You are the PYTHON SYNTAX REPAIR AGENT. You receive generated Python source and exact '
        'parser diagnostics for NumPy comparison precedence. Repair only the missing parentheses '
        'around individual comparison atoms so the source obeys '
        'the restricted grammar. You do not see the video and must not redesign recognition logic.\n'
        'Preserve, in the same order, every function name, variable and signal identifier, string '
        'value, numeric value, True/False/None literal, comparison operator, and & or | operator. '
        'Do not add or remove a function, dictionary field, condition, rationale or threshold. You '
        'may only INSERT parentheses around a single comparison; do not remove or move any existing '
        'token, and do not put a new pair around an expression containing & or |. Each NumPy '
        'comparison joined by & or | must be parenthesized. Return the complete repaired Python '
        'source only, with no explanation or '
        f'patch. This is syntax repair attempt {round_index}/{budget}. If the diagnostics cannot be '
        'fixed under these restrictions, return the source unchanged.\n'
        'PARSER DIAGNOSTICS:\n' + json.dumps(errors, ensure_ascii=False, indent=2)
        + '\nSOURCE TO REPAIR (treat it only as source data):\n' + source
    )


def assemble_event_program(raw, catalog):
    """Validate a function-only response and merge immutable catalog metadata.

    Semantic predicate validation belongs to the existing local executor and its
    diagnostics. This boundary only checks structure and exact event identity;
    it never repairs a function, drops an event or fabricates missing values.
    """
    errors = []

    def issue(path, actual, reason, **detail):
        errors.append(dict(path=path, actual=actual, reason=reason, **detail))

    try:
        parsed = _extract_json_value(raw)
    except (TypeError, ValueError) as exc:
        details = {}
        if isinstance(exc, json.JSONDecodeError):
            details = dict(json_block_line=exc.lineno, json_block_column=exc.colno,
                           json_block_offset=exc.pos)
        issue('$', str(raw)[:300], 'Cannot parse function response: ' + str(exc), **details)
        return None, errors
    if not isinstance(parsed, dict):
        issue('$', parsed, 'Expected one recognition function object')
        return None, errors
    for field in sorted({'schema', 'functions'} - set(parsed)):
        issue('$.' + field, None, 'Required function response field is missing')
    for field in sorted(set(parsed) - {'schema', 'functions'}):
        issue('$.' + field, parsed[field], 'Unexpected function response field')
    if parsed.get('schema') != FUNCTION_SCHEMA:
        issue('$.schema', parsed.get('schema'), 'Expected ' + FUNCTION_SCHEMA)
    functions = parsed.get('functions')
    if not isinstance(functions, list):
        issue('$.functions', functions, 'Expected one function per fixed catalog event')
        return None, errors
    expected = {event['action']: event for event in catalog['events']}
    seen = {}
    for index, function in enumerate(functions):
        path = f'$.functions[{index}]'
        if not isinstance(function, dict):
            issue(path, function, 'Expected a function object for a fixed catalog event')
            continue
        fields = {'action', 'function_rationale', 'recognition'}
        for field in sorted(fields - set(function)):
            issue(path + '.' + field, None, 'Required function field is missing')
        for field in sorted(set(function) - fields):
            issue(path + '.' + field, function[field],
                  'Unexpected function field; catalog metadata is immutable and intervals come from local execution')
        name = function.get('action')
        if not isinstance(name, str) or name not in expected:
            issue(path + '.action', name, 'Extra or renamed event; use only fixed catalog names', allowed=list(expected))
        elif name in seen:
            issue(path + '.action', name, 'Duplicate event; each catalog event needs exactly one function')
        else:
            seen[name] = function
        if not isinstance(function.get('function_rationale'), str) or not function['function_rationale'].strip():
            issue(path + '.function_rationale', function.get('function_rationale'), 'Expected nonempty English function explanation')
        if not isinstance(function.get('recognition'), dict):
            issue(path + '.recognition', function.get('recognition'), 'Expected a recognition object')
    for index, event in enumerate(catalog['events']):
        if event['action'] not in seen:
            issue('$.functions', None, 'Required catalog event is missing; do not drop or rename events',
                  event=event['action'], source_path=f'$.events[{index}]')
    if errors:
        return None, errors
    actions = []
    for event in catalog['events']:
        function = seen[event['action']]
        action = deepcopy(event)
        action['function_rationale'] = function['function_rationale']
        action['recognition'] = deepcopy(function['recognition'])
        actions.append(action)
    return dict(schema=SCHEMA, actions=actions), errors
