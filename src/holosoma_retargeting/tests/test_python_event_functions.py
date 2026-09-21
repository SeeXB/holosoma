import numpy as np
import pytest
from holosoma_retargeting.semantic_keyframes.python_event_functions import (
    parse_python_functions,
    syntax_repairable,
    validate_syntax_only_repair,
)
from holosoma_retargeting.semantic_keyframes.trajectory_events import predicate

CATALOG = {'events': [{'action': 'event_alpha'}]}
SIGNALS = {'speed': np.array([0., .3, .8]), 'height': np.array([1., 2., 3.])}


def source(condition):
    return ('def event_alpha(signals):\n'
            '    return {"function_rationale": "Evidence", "recognition": {"start": '+condition+'}}')


def test_python_array_semantics_survive_compilation():
    text = source('(signals["speed"] > 0.2) & ((signals["height"] <= 2) | (signals["speed"] >= 0.9))')
    parsed, errors = parse_python_functions(text, CATALOG, SIGNALS, 'test')
    assert not errors
    result = predicate(parsed['functions'][0]['recognition']['start'], SIGNALS)
    np.testing.assert_array_equal(result, (SIGNALS['speed'] > .2) & ((SIGNALS['height'] <= 2) | (SIGNALS['speed'] >= .9)))


@pytest.mark.parametrize('condition', [
    '__import__("os").system("false")', 'signals["speed"][0] > 0.1',
    '(signals["speed"] > 0.1) and (signals["height"] < 2)',
    'signals["speed"] > 0.1 & signals["height"] < 2',
    '{"signal": "speed", "op": "gt", "value": 0.1}',
    'signals["unknown"] > 1', 'signals["speed"] > float("nan")',
    'signals["speed"] > True',
])
def test_invalid_python_reports_field_and_source_location(condition):
    parsed, errors = parse_python_functions(source(condition), CATALOG, SIGNALS, 'test')
    assert parsed is None and errors
    assert errors[0]['path'] == 'event_alpha.recognition.start'
    assert errors[0]['python_line'] == 2 and errors[0]['python_column'] > 0


def test_imports_side_effects_and_duplicate_functions_are_not_executed(tmp_path):
    target = tmp_path/'must_not_exist'
    raw = f'open({str(target)!r}, "w").write("unsafe")\n' + source('signals["speed"] > 0.2')
    parsed, errors = parse_python_functions(raw, CATALOG, SIGNALS, 'test')
    assert parsed is None and errors and not target.exists()
    parsed, errors = parse_python_functions(source('signals["speed"] > 0.2')+'\n'+source('signals["speed"] > 0.4'), CATALOG, SIGNALS, 'test')
    assert parsed is None and any('duplicate' in e['reason'] for e in errors)


def test_fenced_python_and_signed_numbers():
    parsed, errors = parse_python_functions('```python\n'+source('signals["speed"] > -0.2')+'\n```', CATALOG, SIGNALS, 'test')
    assert not errors and parsed['functions'][0]['recognition']['start']['value'] == -.2


def test_syntax_error_has_python_line_column():
    parsed, errors = parse_python_functions('def event_alpha(signals)\n    pass', CATALOG, SIGNALS, 'test')
    assert parsed is None and errors[0]['python_line'] == 1 and errors[0]['python_column']


def test_logic_sr_keeps_entire_previous_python_source():
    from generate_multi_agent_event_plans import stage_reflection
    raw = 'def event_alpha(signals):\n    """Embedded JSON: {"note": "not the source"}"""\n    return {}'
    prompt = stage_reflection('Write Python', raw, {'errors': [{'path': 'event_alpha', 'reason': 'missing recognition'}]},
                              role='logic_agent', round_index=1, budget=6, unchanged=False)
    assert raw in prompt and 'missing recognition' in prompt


def test_syntax_only_repair_accepts_parentheses_without_changing_logic_tokens():
    original = source('signals["speed"] > 0.2 | signals["height"] < 2')
    repaired = source('(signals["speed"] > 0.2) | (signals["height"] < 2)')
    _, initial_errors = parse_python_functions(original, CATALOG, SIGNALS, 'test')
    assert syntax_repairable(initial_errors)
    parsed, errors = validate_syntax_only_repair(original, repaired, CATALOG, SIGNALS, 'test')
    assert not errors
    np.testing.assert_array_equal(
        predicate(parsed['functions'][0]['recognition']['start'], SIGNALS),
        (SIGNALS['speed'] > .2) | (SIGNALS['height'] < 2),
    )


@pytest.mark.parametrize('changed', [
    source('(signals["speed"] > 0.3) | (signals["height"] < 2)'),
    source('(signals["speed"] > 0.2) & (signals["height"] < 2)'),
    source('(signals["height"] > 0.2) | (signals["speed"] < 2)'),
])
def test_syntax_only_repair_rejects_threshold_operator_or_signal_changes(changed):
    original = source('signals["speed"] > 0.2 | signals["height"] < 2')
    parsed, errors = validate_syntax_only_repair(original, changed, CATALOG, SIGNALS, 'test')
    assert parsed is None
    assert any('changed protected' in error['reason'] for error in errors)


def test_missing_function_is_not_delegated_as_syntax_repair():
    other_catalog = {'events': [{'action': 'event_alpha'}, {'action': 'event_beta'}]}
    _, errors = parse_python_functions(source('signals["speed"] > 0.2'), other_catalog, SIGNALS, 'test')
    assert not syntax_repairable(errors)


@pytest.mark.parametrize('condition', [
    'False', '__import__("os")', '(signals["speed"] > 0.1) and (signals["height"] < 2)',
])
def test_non_parentheses_failures_are_not_delegated_to_syntax_agent(condition):
    _, errors = parse_python_functions(source(condition), CATALOG, SIGNALS, 'test')
    assert errors and not syntax_repairable(errors)


def test_syntax_repair_cannot_regroup_boolean_expressions():
    original = source(
        'signals["speed"] > 0.2 | signals["height"] < 2 & (signals["speed"] > 0.1)')
    regrouped = source(
        '((signals["speed"] > 0.2) | (signals["height"] < 2)) & (signals["speed"] > 0.1)')
    parsed, errors = validate_syntax_only_repair(original, regrouped, CATALOG, SIGNALS, 'test')
    assert parsed is None
    assert any('regrouped boolean operators' in error['reason'] for error in errors)


def test_semantically_redundant_composite_parentheses_are_allowed():
    original = ('def event_alpha(signals):\n    return {"function_rationale": "Evidence", '
                '"recognition": {"start": signals["speed"] > 0.2 | signals["height"] < 2, '
                '"active": (signals["speed"] > 0.1) & (signals["height"] < 3)}}')
    repaired = ('def event_alpha(signals):\n    return {"function_rationale": "Evidence", '
                '"recognition": {"start": (signals["speed"] > 0.2) | (signals["height"] < 2), '
                '"active": ((signals["speed"] > 0.1) & (signals["height"] < 3))}}')
    parsed, errors = validate_syntax_only_repair(original, repaired, CATALOG, SIGNALS, 'test')
    assert parsed is not None and not errors
