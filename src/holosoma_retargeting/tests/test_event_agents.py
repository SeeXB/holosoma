"""Contracts between visual discovery, function synthesis, and local assembly."""
import json
from copy import deepcopy

import pytest
from holosoma_retargeting.semantic_keyframes.event_agents import (
    EVENT_CATALOG_SCHEMA,
    FUNCTION_SCHEMA,
    assemble_event_program,
    diagnose_event_catalog,
    event_catalog_prompt,
    logic_agent_prompt,
    syntax_repair_agent_prompt,
)


def catalog():
    return dict(schema=EVENT_CATALOG_SCHEMA, events=[dict(
        action='event_alpha', body_parts=['left_hand', 'right_hand'],
        criticality_level=3, rationale='Visible evidence from the original images.')])


def functions():
    return dict(schema=FUNCTION_SCHEMA, functions=[dict(
        action='event_alpha', function_rationale='Conditions identify the visual transition.',
        recognition={'start': {'signal': 'object_speed', 'op': 'gt', 'value': 0.1}})])


def test_catalog_is_validated_without_bilateral_repair():
    source = catalog()
    constraints = dict(mode='bimanual_upper_limbs_v1')
    accepted, report = diagnose_event_catalog(json.dumps(source), constraints)
    assert accepted == source and not report['errors']
    source['events'][0]['body_parts'] = ['right_hand']
    accepted, report = diagnose_event_catalog(json.dumps(source), constraints)
    assert accepted is None
    assert any(row['path'] == '$.events[0].body_parts' and row.get('missing') == 'left_hand'
               for row in report['errors'])
    assert source['events'][0]['body_parts'] == ['right_hand']


@pytest.mark.parametrize('field,value', [
    ('start_frame', 2), ('duration_s', 3), ('recognition', {}), ('function_rationale', 'logic'),
])
def test_catalog_rejects_temporal_and_function_fields(field, value):
    source = catalog()
    source['events'][0][field] = value
    accepted, report = diagnose_event_catalog(json.dumps(source), {})
    assert accepted is None
    assert any(row['path'] == '$.events[0].' + field for row in report['errors'])


def test_catalog_reports_multiple_localized_errors():
    source = catalog()
    source['events'][0].update(body_parts=['left_leg', ['hand']], criticality_level=True, rationale='')
    source['events'].append(deepcopy(source['events'][0]))
    accepted, report = diagnose_event_catalog(json.dumps(source), {})
    assert accepted is None
    paths = {row['path'] for row in report['errors']}
    assert {'$.events[0].body_parts[0]', '$.events[0].body_parts[1]',
            '$.events[0].criticality_level', '$.events[0].rationale', '$.events[1].action'} <= paths


def test_catalog_bad_json_preserves_parse_location():
    accepted, report = diagnose_event_catalog('{"schema":', {})
    assert accepted is None
    assert report['errors'][0]['json_block_line'] == 1
    assert report['errors'][0]['json_block_column'] > 1


@pytest.mark.parametrize('events', [[], [catalog()['events'][0]] * 5, 'events'])
def test_catalog_must_have_one_to_four_events(events):
    source = dict(schema=EVENT_CATALOG_SCHEMA, events=events)
    accepted, report = diagnose_event_catalog(json.dumps(source), {})
    assert accepted is None and report['errors']


def test_prompts_separate_discovery_from_signal_logic():
    visual = event_catalog_prompt(bimanual=True)
    assert '<unique_snake_case_event_name>' in visual
    assert visual.count('<allowed_body_part>') == 1
    assert 'USER-CONFIRMED BIMANUAL' in visual
    assert all(part in visual for part in (
        'pelvis', 'waist', 'left_hip', 'right_knee', 'left_shoulder',
        'right_elbow', 'left_wrist', 'right_hand', 'left_foot', 'right_ankle',
    ))
    assert 'Available signals' not in visual and 'object_vertical_velocity' not in visual
    assert 'object orientation' in visual
    assert not any('\u4e00' <= ch <= '\u9fff' for ch in visual)
    prompt = logic_agent_prompt(catalog(), {'object_speed': 'm/s', 'object_height': 'm'},
                               {'object_speed': {'min': 0.0, 'max': 1.0}}, bimanual=True)
    assert 'Write PYTHON SOURCE CODE' in prompt and 'FIXED EVENT CATALOG' in prompt
    assert 'def EVENT_NAME(signals):' in prompt and 'event_alpha' in prompt
    assert 'Return actions=[]' not in prompt and 'Propose up to' not in prompt
    assert '<finite_number>' in prompt and '<positive_seconds>' in prompt
    assert 'Available signals and units' in prompt
    assert not any('\u4e00' <= ch <= '\u9fff' for ch in prompt)


def test_logic_prompt_explains_object_rotate_without_trajectory_values():
    prompt = logic_agent_prompt(
        catalog(),
        {'object_rotate': 'rad; cumulative unsigned 3-D orientation change from clip start'},
        {'object_rotate': {'min': 0.0, 'max': 4.0}},
    )
    assert 'object_rotate is cumulative unsigned 3-D orientation change' in prompt
    assert 'object_angular_speed is the instantaneous rotation rate' in prompt


def test_syntax_repair_prompt_is_semantics_preserving_and_text_only():
    source = 'def event_alpha(signals):\n    return signals["speed"] > 0.2 | signals["height"] < 2'
    prompt = syntax_repair_agent_prompt(
        source, [{'path': 'event_alpha.recognition.start', 'python_line': 2,
                  'reason': 'comparisons need parentheses'}],
        round_index=1, budget=2)
    assert source in prompt and 'PYTHON SYNTAX REPAIR AGENT' in prompt
    assert 'must not redesign recognition logic' in prompt
    assert 'Do not add or remove a function' in prompt
    assert 'images' not in prompt.lower()


def test_assembly_preserves_catalog_metadata_and_order_without_mutation():
    source = catalog()
    source['events'].append(dict(action='event_beta', body_parts=['pelvis'],
                                 criticality_level=2, rationale='Other visual evidence.'))
    response = functions()
    response['functions'].insert(0, dict(action='event_beta', function_rationale='Other conditions.', recognition={}))
    original = deepcopy(source)
    program, errors = assemble_event_program(json.dumps(response), source)
    assert not errors
    assert [action['action'] for action in program['actions']] == ['event_alpha', 'event_beta']
    for event, action in zip(source['events'], program['actions']):
        assert all(action[key] == value for key, value in event.items())
    program['actions'][0]['body_parts'].append('pelvis')
    assert source == original


@pytest.mark.parametrize('change,expected_path', [
    ('drop', '$.functions'), ('rename', '$.functions[0].action'),
    ('duplicate', '$.functions[1].action'), ('metadata', '$.functions[0].body_parts'),
    ('timing', '$.functions[0].start_frame'), ('wrong_schema', '$.schema'),
])
def test_assembly_rejects_missing_extra_or_changed_events(change, expected_path):
    response = functions()
    if change == 'drop':
        response['functions'] = []
    elif change == 'rename':
        response['functions'][0]['action'] = 'different_event'
    elif change == 'duplicate':
        response['functions'].append(deepcopy(response['functions'][0]))
    elif change == 'metadata':
        response['functions'][0]['body_parts'] = ['right_hand']
    elif change == 'timing':
        response['functions'][0]['start_frame'] = 0
    elif change == 'wrong_schema':
        response['schema'] = 'wrong'
    program, errors = assemble_event_program(json.dumps(response), catalog())
    assert program is None
    assert expected_path in {row['path'] for row in errors}


def test_assembly_never_fabricates_missing_function_fields():
    response = functions()
    del response['functions'][0]['recognition']
    program, errors = assemble_event_program(json.dumps(response), catalog())
    assert program is None
    assert any(row['path'] == '$.functions[0].recognition' for row in errors)
