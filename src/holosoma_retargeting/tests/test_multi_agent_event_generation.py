"""Stage separation, bounded current-run SR and deterministic aggregation."""
# ruff: noqa: F401, F811 -- importing this fixture makes it available to pytest.
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from test_trajectory_events import fresh_runner  # shared fresh-source fixture


def stage_outputs(plan):
    from holosoma_retargeting.semantic_keyframes.event_agents import EVENT_CATALOG_SCHEMA, FUNCTION_SCHEMA
    catalog = dict(schema=EVENT_CATALOG_SCHEMA, events=[
        {k: v for k, v in action.items() if k in ('action', 'body_parts', 'criticality_level', 'rationale')}
        for action in plan['actions']])
    functions = dict(schema=FUNCTION_SCHEMA, functions=[
        {k: v for k, v in action.items() if k in ('action', 'function_rationale', 'recognition')}
        for action in plan['actions']])
    return catalog, functions


def python_source(functions):
    def expression(value):
        if 'signal' in value:
            ops = dict(gt='>', ge='>=', lt='<', le='<=')
            return f"(signals[{value['signal']!r}] {ops[value['op']]} {value['value']!r})"
        op = ' & ' if 'all' in value else ' | '
        return '(' + op.join(expression(v) for v in next(iter(value.values()))) + ')'
    lines = []
    for function in functions['functions']:
        lines += [f"def {function['action']}(signals):", '    return {',
                  f"        'function_rationale': {function['function_rationale']!r},", "        'recognition': {"]
        for key, value in function['recognition'].items():
            rendered = expression(value) if key in ('start', 'active', 'end', 'contact') and value is not None else repr(value)
            lines.append(f"            {key!r}: {rendered},")
        lines += ['        },', '    }']
    return '\n'.join(lines)


def test_agents_use_separate_contexts_then_replay_intervals(fresh_runner, monkeypatch, tmp_path):
    import generate_multi_agent_event_plans as multi
    from holosoma_retargeting.semantic_keyframes.trajectory_events import execute_event_program
    from stage_fresh_event_plans import stage
    _, plan, args, _ = fresh_runner
    args['sr_rounds'] = 6
    catalog, functions = stage_outputs(plan)
    prompts = []
    def vlm(images, prompt):
        assert images and 'OLD_RUN_SENTINEL' not in prompt
        prompts.append((images, prompt))
        if len(prompts) <= 2:
            assert 'Trajectory signal ranges' not in prompt
            return json.dumps(catalog)
        assert all(json.dumps(part) in prompt for part in catalog['events'][0]['body_parts'])
        return python_source(functions)
    monkeypatch.setattr(multi, 'call_vlm', vlm)
    args['output_root'].mkdir()
    (args['output_root']/'event_catalog.json').write_text('OLD_RUN_SENTINEL')
    status = multi.generate_once(**args)
    assert status['status'] == 'requires_visual_and_signal_review', status
    assert status['agent_calls'] == {'event_agent': 2, 'logic_agent': 2}
    assert status['reflection_round'] == 2 and status['vlm_calls'] == 4
    assert all(images == prompts[0][0] for images, _ in prompts)
    output, audit = Path(status['output']), Path(status['audit'])
    program = json.loads((output/'event_program.json').read_text())
    assert program == plan
    with np.load(output/'trajectory_signals.npz') as data:
        replay = execute_event_program(program, {k: data[k] for k in data if k != 'fps'}, float(data['fps']))
    assert replay['coverage'] == status['coverage']
    assert replay['events'][0]['windows'] == [dict(start_frame=8, trigger_frame=8, end_frame=17)]
    manifest = json.loads((audit/'manifest.json').read_text())
    assert 'event_catalog.json' in manifest['input_sha256']
    assert 'event_agents.py' in ' '.join(manifest['code_sha256'])
    assert manifest['max_vlm_calls'] == 9
    assert manifest['agents'] == ['event_agent', 'logic_agent', 'syntax_repair_agent']
    row = dict(dataset='omomo', task='test_task', **status)
    stage(row, tmp_path/'staged', 'Synthetic integration fixture')
    functions_source = (output/'event_functions.py').read_text()
    (output/'event_functions.py').write_text(functions_source + '\n# tampered\n')
    with pytest.raises(ValueError, match='function source changed'):
        stage(row, tmp_path/'must_fail_source', 'Tampered source')
    (output/'event_functions.py').write_text(functions_source)
    (output/'event_catalog.json').write_text('{}')
    with pytest.raises(ValueError, match='Prepared input changed'):
        stage(row, tmp_path/'must_fail', 'Tampered catalog')


def test_agents_share_six_reflections_not_six_per_agent(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, plan, args, _ = fresh_runner
    args['sr_rounds'] = 6
    catalog, functions = stage_outputs(plan)
    prompts = []
    def vlm(images, prompt):
        prompts.append(prompt)
        if len(prompts) <= 2:
            return json.dumps(catalog)
        if len(prompts) > 3:
            assert f'unknown_signal_{len(prompts)-1}' in prompt
            assert 'lift.recognition.start' in prompt
            if len(prompts) > 4:
                assert f'unknown_signal_{len(prompts)-2}' not in prompt
        bad = deepcopy(functions)
        bad['functions'][0]['recognition']['start']['signal'] = f'unknown_signal_{len(prompts)}'
        return python_source(bad)
    monkeypatch.setattr(multi, 'call_vlm', vlm)
    status = multi.generate_once(**args)
    assert status['status'] == 'logic_agent_validation_failed', status
    assert status['vlm_calls'] == 8 and status['reflection_round'] == 6
    assert status['agent_calls'] == {'event_agent': 2, 'logic_agent': 6}
    assert not (Path(status['output'])/'semantic_plan.candidate.json').exists()


def test_invalid_event_agent_blocks_logic_agent(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, plan, args, _ = fresh_runner
    args['sr_rounds'] = 6
    catalog, _ = stage_outputs(plan)
    catalog['events'][0]['start_frame'] = 0
    monkeypatch.setattr(multi, 'call_vlm', lambda *args: json.dumps(catalog))
    status = multi.generate_once(**args)
    assert status['status'] == 'event_agent_validation_failed', status
    assert status['agent_calls'] == {'event_agent': 3}
    assert 'logic_agent' not in status['agent_calls']
    assert not (Path(status['output'])/'event_program.json').exists()


def test_prepare_only_calls_neither_agent(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, _, args, _ = fresh_runner
    args['prepare_only'] = True
    monkeypatch.setattr(multi, 'call_vlm', lambda *args: pytest.fail('No network in prepare-only'))
    status = multi.generate_once(**args)
    assert status['status'] == 'prepared_only' and status['vlm_calls'] == 0
    assert not (Path(status['output'])/'event_catalog.json').exists()


def test_untriggered_logic_cannot_be_aggregated_into_fake_windows(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, plan, args, _ = fresh_runner
    args['sr_rounds'] = 6
    catalog, functions = stage_outputs(plan)
    functions['functions'][0]['recognition']['start']['value'] = 100
    count = 0
    def vlm(*args):
        nonlocal count
        count += 1
        return json.dumps(catalog) if count <= 2 else python_source(functions)
    monkeypatch.setattr(multi, 'call_vlm', vlm)
    status = multi.generate_once(**args)
    assert status['status'] == 'unresolved_functions', status
    candidate = json.loads((Path(status['output'])/'semantic_plan.candidate.json').read_text())
    assert candidate['events'] == [] and candidate['unresolved_events']
    assert candidate['coverage']['key_frames'] == 0


def test_text_only_syntax_agent_repairs_parentheses_before_execution(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, plan, args, _ = fresh_runner
    args.update(sr_rounds=6, syntax_repair_rounds=1)
    catalog, functions = stage_outputs(plan)
    recognition = functions['functions'][0]['recognition']
    recognition['active'] = {'all': [deepcopy(recognition['active']), deepcopy(recognition['active'])]}
    valid = python_source(functions)
    comparison = "(signals['object_speed'] > 0.2)"
    bad = valid.replace(f'({comparison} & {comparison})',
                        "(signals['object_speed'] > 0.2 & signals['object_speed'] > 0.2)")
    assert bad != valid
    calls = []

    def vlm(images, prompt):
        calls.append((images, prompt))
        if 'VIDEO EVENT AGENT' in prompt:
            return json.dumps(catalog)
        if 'PYTHON SYNTAX REPAIR AGENT' in prompt:
            assert images == []
            assert bad in prompt and 'python_line' in prompt
            return valid
        return bad if sum('TRAJECTORY LOGIC AGENT' in text for _, text in calls) == 1 else valid

    monkeypatch.setattr(multi, 'call_vlm', vlm)
    status = multi.generate_once(**args)
    assert status['status'] == 'requires_visual_and_signal_review', status
    assert status['agent_calls'] == {
        'event_agent': 2, 'logic_agent': 1, 'syntax_repair_agent': 1,
    }
    assert status['syntax_repair_calls'] == 1 and status['vlm_calls'] == 4
    output, audit = Path(status['output']), Path(status['audit'])
    assert (output/'event_functions.py').read_text() == valid
    syntax_requests = [json.loads(path.read_text()) for path in output.glob('request_*.json')
                       if 'PYTHON SYNTAX REPAIR AGENT' in path.read_text()]
    assert len(syntax_requests) == 1
    assert len(syntax_requests[0]['messages'][0]['content']) == 1
    validations = [json.loads(path.read_text()) for path in audit.glob('validation_*.json')]
    assert any(row['agent'] == 'syntax_repair_agent' and row['valid']
               and row['protected_tokens_unchanged'] for row in validations)


def test_syntax_agent_cannot_change_threshold_to_make_program_pass(fresh_runner, monkeypatch):
    import generate_multi_agent_event_plans as multi
    _, plan, args, _ = fresh_runner
    args.update(sr_rounds=0, syntax_repair_rounds=1)
    catalog, functions = stage_outputs(plan)
    recognition = functions['functions'][0]['recognition']
    recognition['active'] = {'all': [deepcopy(recognition['active']), deepcopy(recognition['active'])]}
    valid = python_source(functions)
    comparison = "(signals['object_speed'] > 0.2)"
    bad = valid.replace(f'({comparison} & {comparison})',
                        "(signals['object_speed'] > 0.2 & signals['object_speed'] > 0.2)")
    changed_threshold = valid.replace('> 0.2', '> 0.3')
    calls = []

    def vlm(images, prompt):
        calls.append(prompt)
        if 'VIDEO EVENT AGENT' in prompt:
            return json.dumps(catalog)
        if 'PYTHON SYNTAX REPAIR AGENT' in prompt:
            return changed_threshold
        return bad

    monkeypatch.setattr(multi, 'call_vlm', vlm)
    status = multi.generate_once(**args)
    assert status['status'] == 'logic_agent_validation_failed', status
    assert status['agent_calls'] == {
        'event_agent': 1, 'logic_agent': 1, 'syntax_repair_agent': 1,
    }
    assert not (Path(status['output'])/'event_program.json').exists()
    validations = [json.loads(path.read_text()) for path in Path(status['audit']).glob('validation_*.json')]
    repair = next(row for row in validations if row['agent'] == 'syntax_repair_agent')
    assert not repair['valid'] and not repair['protected_tokens_unchanged']
