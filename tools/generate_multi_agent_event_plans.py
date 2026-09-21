#!/usr/bin/env python3
"""Visual event, logic and guarded syntax-repair agents with local execution."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from holosoma_retargeting.semantic_keyframes.event_agents import (
    FUNCTION_SCHEMA,
    assemble_event_program,
    diagnose_event_catalog,
    event_catalog_prompt,
    logic_agent_prompt,
    syntax_repair_agent_prompt,
)
from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
from holosoma_retargeting.semantic_keyframes.pipeline import _extract_json_value, call_vlm
from holosoma_retargeting.semantic_keyframes.python_event_functions import (
    parse_python_functions,
    syntax_repairable,
    validate_syntax_only_repair,
)
from holosoma_retargeting.semantic_keyframes.trajectory_events import execute_event_program
from pilot_trajectory_event_plans import (
    AUTHORIZED_HOST,
    ROOT,
    file_hash,
    write_json,
)
from pilot_trajectory_event_plans import (
    generate_once as prepare_run,
)

VERSION = 'trajectory_event_agents_v3_object_rotate_20260921'
WORKFLOW = 'visual_event_agent_then_logic_agent_then_syntax_repair_agent_then_trajectory_executor'


def stage_reflection(prompt, raw, feedback, *, role, round_index, budget, unchanged):
    """Current stage's immediate predecessor only; no previous-run or other-stage critique."""
    if role == 'logic_agent':
        previous = raw
    else:
        try:
            previous = json.dumps(_extract_json_value(raw), indent=2, ensure_ascii=False)
        except (ValueError, TypeError):
            previous = raw
    return (
        prompt + f'\n\nSELF-REFLECTION: {role}, shared SR pass {round_index}/{budget}. '
        'Review the same original whole-clip images, the immediately previous output below, and '
        'its specific errors. Correct each reported field and reason. Return the COMPLETE revised '
        'output in this stage\'s SAME format (Python source for the logic agent). Do not add critique fields or copy diagnostic frame '
        'indices into the output. Diagnostics are observations, not target windows. '
        'Do not delete observed events or relax thresholds merely to pass validation. '
        + ('The previous output was unchanged and its problems remain. Reconsider the causes. '
           if unchanged else '')
        + '\nImmediately previous output from this stage and THIS RUN:\n' + previous
        + '\nError locations, reasons and local evidence:\n' + json.dumps(feedback, ensure_ascii=False)
    )


def logic_feedback(raw, catalog, constraints, signals, fps):
    """Join immutable event metadata, execute functions, and point errors to function fields."""
    parsed, errors = parse_python_functions(raw, catalog, signals, FUNCTION_SCHEMA)
    if errors:
        return None, dict(errors=errors, warnings=[], per_event=[])
    program, errors = assemble_event_program(json.dumps(parsed), catalog)
    if errors:
        return None, dict(errors=errors, warnings=[], per_event=[])
    diagnostics = diagnose_response(json.dumps(program), constraints, signals, fps)

    def remap(value):
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                if key in ('path', 'conflicts_with') and isinstance(item, str):
                    for i, event in enumerate(catalog['events']):
                        prefix = f'$.actions[{i}]'
                        if item == prefix or item.startswith(prefix + '.'):
                            suffix = item[len(prefix):]
                            # Frozen event metadata is visible in the catalog, not editable by this agent.
                            if suffix.startswith(('.body_parts', '.rationale', '.criticality_level')):
                                item = f'event_catalog.events[{i}]' + suffix
                            else:
                                item = event["action"] + suffix
                            break
                    result[key] = item
                else:
                    result[key] = remap(item)
            return result
        if isinstance(value, list):
            return [remap(item) for item in value]
        return value

    diagnostics = remap(diagnostics)
    if constraints:
        program['body_part_constraints'] = constraints
    if diagnostics['errors']:
        return None, diagnostics
    result = execute_event_program(program, signals, fps)
    diagnostics.update(coverage=result['coverage'], unresolved_events=result['unresolved_events'])
    return (program, result), diagnostics


def generate_once(*, bundle, video, constraints, output_root, audit_root, samples=40,
                  prepare_only=False, lafan_source=None, sr_rounds=6, syntax_repair_rounds=1):
    if type(syntax_repair_rounds) is not int or not 0 <= syntax_repair_rounds <= 3:
        raise ValueError('syntax_repair_rounds must be an integer between 0 and 3')
    # Reuse fresh extraction and source hashing, never its monolithic model generation.
    status = prepare_run(bundle=bundle, video=video, constraints=constraints,
                         output_root=output_root, audit_root=audit_root, samples=samples,
                         prepare_only=True, lafan_source=lafan_source, sr_rounds=sr_rounds)
    if status['status'] != 'prepared_only':
        return status
    output, audit = Path(status['output']), Path(status['audit'])
    status.update(workflow=WORKFLOW, agent='event_agent', reflection_round=0,
                  syntax_repair_calls=0, agent_calls={}, max_self_reflection_rounds=sr_rounds,
                  max_syntax_repair_calls=syntax_repair_rounds)
    stage = 'preparation'
    try:
        request = json.loads((output/'request.json').read_text())
        images = request['messages'][0]['content'][1:]
        units = json.loads((output/'signal_units.json').read_text())
        with np.load(output/'trajectory_signals.npz', allow_pickle=False) as data:
            fps = float(data['fps'])
            signals = {k: data[k].copy() for k in data if k != 'fps'}
        ranges = {k: dict(min=float(v.min()), max=float(v.max())) for k, v in signals.items()}
        prompt = event_catalog_prompt(bimanual=constraints.get('mode') == 'bimanual_upper_limbs_v1',
                                      object_task='object_height' in units)
        request['messages'] = [dict(role='user', content=[dict(type='text', text=prompt), *images])]
        write_json(output/'request.json', request)
        (audit/'prompt.txt').write_text(prompt)
        manifest = json.loads((audit/'manifest.json').read_text())
        for path in (Path(__file__).resolve(), ROOT/'src/holosoma_retargeting/holosoma_retargeting/semantic_keyframes/event_agents.py',
                     ROOT/'src/holosoma_retargeting/holosoma_retargeting/semantic_keyframes/python_event_functions.py'):
            relative = path.relative_to(ROOT)
            target = audit/'code_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            manifest['code_sha256'][str(relative)] = file_hash(target)
        manifest.update(workflow=WORKFLOW,
                        agents=['event_agent', 'logic_agent', 'syntax_repair_agent'],
                        aggregation='deterministic trajectory execution; no LLM interval editing',
                        logic_language='restricted_python_ast_v1',
                        reflection_budget=('shared across event and logic agents, at most six SR calls plus '
                                           'two initial calls'),
                        syntax_repair_policy=('text-only; parser-triggered; protected tokens must remain unchanged; '
                                              'no images, new functions, signals, thresholds or operators'),
                        max_syntax_repair_calls=syntax_repair_rounds,
                        max_vlm_calls=2+sr_rounds+syntax_repair_rounds,
                        prompt_sha256=file_hash(audit/'prompt.txt'),
                        input_sha256={str(p.relative_to(output)): file_hash(p)
                                      for p in sorted(output.rglob('*')) if p.is_file()})
        write_json(audit/'manifest.json', manifest)
        if prepare_only:
            write_json(audit/'status.json', status)
            return status
        endpoint = urlsplit(os.environ.get('OPENAI_BASE_URL', ''))
        if endpoint.scheme != 'https' or endpoint.hostname != AUTHORIZED_HOST or endpoint.username or endpoint.password:
            raise ValueError('Unexpected VLM endpoint')
        if not request['model'] or not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('VLM model and credentials must be configured')

        def request_agent(role, prompt, attached_images, agent_round):
            nonlocal stage
            stage = 'request'
            call_index = status['vlm_calls']
            status.update(status='requesting', agent=role, agent_round=agent_round,
                          vlm_calls=call_index+1)
            status['agent_calls'][role] = status['agent_calls'].get(role, 0)+1
            write_json(audit/'status.json', status)
            round_request = dict(request, messages=[dict(role='user', content=[
                dict(type='text', text=prompt), *attached_images])])
            write_json(output/f'request_{call_index}.json', round_request)
            (audit/f'prompt_{call_index}.txt').write_text(prompt)
            raw = call_vlm(attached_images, prompt)
            (audit/f'response_{call_index}.txt').write_text(raw)
            if role == 'logic_agent':
                (audit/f'logic_response_{call_index}.py').write_text(raw)
            elif role == 'syntax_repair_agent':
                (audit/f'syntax_repair_response_{call_index}.py').write_text(raw)
            return call_index, raw

        def repair_syntax(raw, parse_errors):
            nonlocal stage
            if (not syntax_repairable(parse_errors)
                    or status['syntax_repair_calls'] >= syntax_repair_rounds):
                return raw, parse_errors
            original = raw
            errors = parse_errors
            while status['syntax_repair_calls'] < syntax_repair_rounds:
                repair_round = status['syntax_repair_calls'] + 1
                repair_prompt = syntax_repair_agent_prompt(
                    original, errors, round_index=repair_round, budget=syntax_repair_rounds)
                call_index, candidate = request_agent(
                    'syntax_repair_agent', repair_prompt, [], repair_round-1)
                status['syntax_repair_calls'] += 1
                stage = 'syntax_repair_agent_validation'
                parsed, repair_errors = validate_syntax_only_repair(
                    original, candidate, catalog, signals, FUNCTION_SCHEMA)
                write_json(audit/f'validation_{call_index}.json', dict(
                    agent='syntax_repair_agent', agent_round=repair_round-1,
                    valid=parsed is not None and not repair_errors,
                    repair_scope='atomic_comparison_parentheses_only',
                    input_sha256=hashlib.sha256(original.encode()).hexdigest(),
                    output_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                    protected_tokens_unchanged=not any(
                        'changed protected' in error.get('reason', '')
                        or 'regrouped boolean' in error.get('reason', '')
                        for error in repair_errors),
                    errors=repair_errors, warnings=[], per_event=[]))
                if parsed is not None and not repair_errors:
                    return candidate, []
                errors = repair_errors
            return raw, parse_errors + [dict(
                path='$', reason='Syntax repair budget exhausted without an invariant-preserving valid program')]

        def validate_logic(raw):
            parsed, parse_errors = parse_python_functions(
                raw, catalog, signals, FUNCTION_SCHEMA)
            effective = raw
            if parse_errors:
                effective, remaining = repair_syntax(raw, parse_errors)
                if remaining:
                    return None, dict(errors=remaining, warnings=[], per_event=[]), effective
            value, feedback = logic_feedback(effective, catalog, constraints, signals, fps)
            return value, feedback, effective

        def run_agent(role, base_prompt, validate):
            nonlocal stage
            current_prompt, previous, same_count = base_prompt, None, 0
            local_round = 0
            while True:
                call_index, raw = request_agent(role, current_prompt, images, local_round)
                stage = role + '_validation'
                validated = validate(raw)
                if len(validated) == 3:
                    value, feedback, effective = validated
                else:
                    value, feedback = validated
                    effective = raw
                stage = role + '_validation'
                valid = value is not None and not feedback.get('errors')
                write_json(audit/f'validation_{call_index}.json', dict(
                    agent=role, agent_round=local_round, sr_used=status['reflection_round'],
                    valid=valid, **feedback))
                try:
                    normalized = effective if role == 'logic_agent' else _extract_json_value(effective)
                except (ValueError, TypeError):
                    normalized = effective
                fingerprint = json.dumps([normalized, feedback], sort_keys=True)
                same_count = same_count+1 if fingerprint == previous else 1
                healthy = valid and not feedback.get('warnings') and not feedback.get('unresolved_events')
                # A separately validated syntax repair is already a second review of this source.
                # Otherwise each semantic agent gets one follow-up while the shared budget remains.
                if healthy and (effective != raw or local_round > 0
                                or status['reflection_round'] >= sr_rounds):
                    return value, feedback, effective
                if same_count >= 3 or status['reflection_round'] >= sr_rounds:
                    status['stop_reason'] = ('three_identical_outputs_and_diagnostics' if same_count >= 3
                                             else 'shared_reflection_budget_exhausted')
                    if not valid:
                        raise ValueError(json.dumps(feedback['errors'], ensure_ascii=False))
                    return value, feedback, effective
                status['reflection_round'] += 1
                current_prompt = stage_reflection(base_prompt, effective, feedback, role=role,
                    round_index=status['reflection_round'], budget=sr_rounds, unchanged=fingerprint == previous)
                previous = fingerprint
                local_round += 1

        catalog, _, _ = run_agent('event_agent', prompt, lambda raw: diagnose_event_catalog(raw, constraints))
        write_json(output/'event_catalog.json', catalog)
        # Freeze this stage handoff and validate it again when staging for retargeting.
        manifest['input_sha256']['event_catalog.json'] = file_hash(output/'event_catalog.json')
        write_json(audit/'manifest.json', manifest)
        logic_prompt = logic_agent_prompt(catalog, units, ranges,
                                         bimanual=constraints.get('mode') == 'bimanual_upper_limbs_v1')
        value, feedback, logic_source = run_agent('logic_agent', logic_prompt, validate_logic)
        program, result = value
        (output/'event_functions.py').write_text(logic_source)
        result['candidate_status'] = ('no_event_proposal' if not program['actions'] else
            'unresolved_functions' if result['unresolved_events'] else
            'semantic_review_required' if feedback['errors'] or feedback['warnings'] else
            'requires_visual_and_signal_review')
        result['generation_metadata'] = dict(run_id=status['run_id'], manifest=str(audit/'manifest.json'),
            workflow=WORKFLOW, generation_policy=manifest['generation_policy'],
            self_reflection_rounds=status['reflection_round'], agent_calls=status['agent_calls'],
            syntax_repair_calls=status['syntax_repair_calls'],
            event_functions_sha256=file_hash(output/'event_functions.py'),
            event_catalog_sha256=file_hash(output/'event_catalog.json'),
            body_part_constraints=constraints, review_only=True, active_for_retargeting=False)
        write_json(output/'event_program.json', program)
        write_json(output/'semantic_plan.candidate.json', result)
        write_json(audit/'resolution.json', result)
        write_json(audit/'aggregation.json', dict(
            event_count=len(catalog['events']), localized_events=len(result['events']),
            unresolved_events=result['unresolved_events'], coverage=result['coverage'],
            boundary_source='trajectory_predicates', review_only=True))
        status.update(status=result['candidate_status'], coverage=result['coverage'], agent='local_executor')
    except Exception as exc:
        status.update(status=stage+'_failed', error_type=type(exc).__name__)
        if getattr(exc, 'provider_error', None):
            status['provider_error'] = exc.provider_error
        if stage != 'request':
            status['error'] = str(exc)
    write_json(audit/'status.json', status)
    return status
