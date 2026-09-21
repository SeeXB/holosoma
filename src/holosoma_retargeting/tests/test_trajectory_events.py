from copy import deepcopy

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes.trajectory_events import SCHEMA, execute_event_program


def example():
    signals = {'object_height': np.zeros(30), 'speed': np.r_[np.zeros(3), np.ones(22), np.zeros(5)],
               'left_hand_surface_distance': np.r_[np.ones(8), np.zeros(10), np.ones(12)]}
    rule = lambda signal, op, value: dict(signal=signal, op=op, value=value)
    plan = dict(schema=SCHEMA, actions=[dict(action='lift', body_parts=['left_hand'], criticality_level=3,
                rationale='Both hands lift the object.', function_rationale='Contact plus upward motion.',
                recognition=dict(start=rule('speed','gt',0.5), active=rule('speed','gt',0.2),
                                 end=rule('speed','lt',0.1), start_hold_s=0.2, end_hold_s=0.2,
                                 min_duration_s=0.2, max_duration_s=2, allow_initial=False, initial_evidence='',
                                 contact=rule('left_hand_surface_distance','lt',0.05),
                                 contact_timing='current', contact_lookback_s=0.2))])
    return plan, signals


def test_contact_gate_leaves_preparation_and_release_ordinary():
    plan, signals = example()
    result = execute_event_program(plan, signals, 10)
    assert result['events'][0]['windows'] == [dict(start_frame=8, trigger_frame=8, end_frame=17)]
    assert result['coverage']['ordinary_frames'] == 20


def test_no_contact_never_falls_back_to_nearest_or_zero():
    plan, signals = example()
    signals['left_hand_surface_distance'][:] = 0.2
    result = execute_event_program(plan, signals, 10)
    assert not result['events'] and result['unresolved_events']
    assert result['coverage']['key_frames'] == 0


def test_repeated_events_remain_separate_with_gaps():
    plan, signals = example()
    signals['left_hand_surface_distance'][:] = 1
    signals['left_hand_surface_distance'][5:10] = 0
    signals['left_hand_surface_distance'][15:20] = 0
    result = execute_event_program(plan, signals, 10)
    assert [w['start_frame'] for w in result['events'][0]['windows']] == [5,15]
    assert result['coverage']['key_frames'] == 10


def test_initial_condition_requires_explicit_initial_event():
    plan, signals = example()
    signals['speed'][:3] = 1
    signals['left_hand_surface_distance'][:8] = 0
    assert not execute_event_program(plan, signals, 10)['events']
    plan['actions'][0]['recognition'].update(allow_initial=True, initial_evidence='Already grasping at video start.')
    assert execute_event_program(plan, signals, 10)['events'][0]['windows'][0]['start_frame'] == 0


def test_overlong_or_unterminated_intervals_are_not_clipped_into_success():
    plan, signals = example()
    plan['actions'][0]['recognition']['max_duration_s'] = 0.5
    assert not execute_event_program(plan, signals, 10)['events']
    plan['actions'][0]['recognition']['max_duration_s'] = 5
    signals['speed'][25:] = 1
    signals['left_hand_surface_distance'][18:] = 0
    result = execute_event_program(plan, signals, 10)
    assert not result['events']
    assert result['unresolved_events'][0]['rejected'][0]['reason'] == 'termination_not_observed'


def test_center_distance_and_missing_contact_gate_are_rejected():
    plan, signals = example()
    bad = deepcopy(plan)
    bad['actions'][0]['recognition']['contact']['signal'] = 'object_height'
    with pytest.raises(ValueError, match='surface proximity'):
        execute_event_program(bad, signals, 10)
    plan['actions'][0]['recognition'].update(contact=None, contact_timing='none')
    with pytest.raises(ValueError, match='requires a surface contact gate'):
        execute_event_program(plan, signals, 10)


def test_visual_frame_anchors_are_rejected():
    plan, signals = example()
    plan['actions'][0]['start_frame'] = 0
    with pytest.raises(ValueError, match='no visual timing'):
        execute_event_program(plan, signals, 10)


@pytest.fixture
def fresh_runner(tmp_path, monkeypatch):
    import pilot_trajectory_event_plans as pilot
    plan, signals = example()
    signals['object_speed'] = signals.pop('speed')
    for field in ('start', 'active', 'end'):
        plan['actions'][0]['recognition'][field]['signal'] = 'object_speed'
    bundle, video = tmp_path / 'source.npz', tmp_path / 'source.mp4'
    bundle.write_bytes(b'original trajectory')
    video.write_bytes(b'original video')
    monkeypatch.setattr(pilot, 'load_event_signals', lambda _: (signals, {k:'m' for k in signals}, 10))
    image_dirs = []

    def images(video_path, frames, fps, directory):
        assert video_path == video
        assert frames[0] == 0 and frames[-1] == 29
        assert not directory.exists()
        directory.mkdir()
        (directory / 'frame.jpg').write_bytes(b'fresh image')
        image_dirs.append(directory)
        return [dict(type='image_url', image_url=dict(url='data:image/jpeg;base64,ZnJlc2g='))]

    monkeypatch.setattr(pilot, 'video_images', images)
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://' + pilot.AUTHORIZED_HOST + '/v1')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    monkeypatch.setenv('OPENAI_API_KEY', 'SECRET_MUST_NOT_BE_ARCHIVED')
    args = dict(bundle=bundle, video=video, constraints={}, output_root=tmp_path / 'outputs',
                audit_root=tmp_path / 'audit', samples=10, sr_rounds=0)
    return pilot, plan, args, image_dirs


def test_each_generation_has_fresh_visual_context_and_no_history(fresh_runner, monkeypatch):
    import json
    from pathlib import Path
    pilot, plan, args, image_dirs = fresh_runner
    calls = []

    def vlm(images, prompt):
        assert images
        assert 'OLD_RESPONSE_SENTINEL' not in prompt
        calls.append((images, prompt))
        return json.dumps(plan)

    monkeypatch.setattr(pilot, 'call_vlm', vlm)
    args['output_root'].mkdir()
    (args['output_root'] / 'event_program.json').write_text('OLD_RESPONSE_SENTINEL')
    first = pilot.generate_once(**args)
    (Path(first['audit']) / 'response.txt').write_text('OLD_RESPONSE_SENTINEL')
    second = pilot.generate_once(**args)
    assert len(calls) == 2 and calls[0] == calls[1]
    assert image_dirs[0] != image_dirs[1]
    assert first['run_id'] != second['run_id']
    for run in (first, second):
        assert run['status'] == 'requires_visual_and_signal_review'
        assert run['coverage']['ordinary_frames'] == 20
        assert not run['active_for_retargeting']
        manifest = json.loads((Path(run['audit']) / 'manifest.json').read_text())
        assert manifest['source_sha256'] and manifest['code_sha256']
        assert manifest['generation_policy'] == 'fresh_single_request_no_history_no_feedback'
        request = (Path(run['output']) / 'request.json').read_text()
        assert 'SECRET_MUST_NOT_BE_ARCHIVED' not in request
        assert len(json.loads(request)['messages']) == 1
        # Offline trajectory execution reproduces boundaries without any VLM call.
        stored = json.loads((Path(run['output']) / 'event_program.json').read_text())
        with np.load(Path(run['output']) / 'trajectory_signals.npz') as data:
            replay = execute_event_program(stored, {k:data[k] for k in data if k != 'fps'}, float(data['fps']))
        assert replay['coverage'] == run['coverage']
        assert replay['events'][0]['windows'] == [dict(start_frame=8, trigger_frame=8, end_frame=17)]


@pytest.mark.parametrize('reply', ['not JSON', '{"schema":"wrong","actions":[]}'])
def test_failed_generation_never_repairs_or_fabricates(fresh_runner, monkeypatch, reply):
    from pathlib import Path
    pilot, _, args, _ = fresh_runner
    calls = []
    monkeypatch.setattr(pilot, 'call_vlm', lambda images, prompt: calls.append((images, prompt)) or reply)
    status = pilot.generate_once(**args)
    assert len(calls) == 1
    assert status['status'] == 'validation_failed'
    assert not (Path(status['output']) / 'semantic_plan.candidate.json').exists()
    assert (Path(status['audit']) / 'response.txt').read_text() == reply


def test_prepare_only_never_sends(fresh_runner, monkeypatch):
    pilot, _, args, _ = fresh_runner
    monkeypatch.setattr(pilot, 'call_vlm', lambda *_: pytest.fail('prepare-only sent a request'))
    status = pilot.generate_once(**args, prepare_only=True)
    assert status['status'] == 'prepared_only' and status['vlm_calls'] == 0


def test_prepare_exposes_object_rotate_to_the_planning_agents(fresh_runner, monkeypatch):
    from pathlib import Path
    pilot, _, args, _ = fresh_runner
    signals = {
        'object_height': np.zeros(4),
        'object_rotate': np.arange(4, dtype=float),
        'left_hand_surface_distance': np.ones(4),
    }
    units = {key: 'rad' if key == 'object_rotate' else 'm' for key in signals}
    monkeypatch.setattr(pilot, 'load_event_signals', lambda _: (signals, units, 10.0))

    status = pilot.generate_once(**args, prepare_only=True)

    with np.load(Path(status['output']) / 'trajectory_signals.npz') as stored:
        np.testing.assert_array_equal(stored['object_rotate'], signals['object_rotate'])


def test_source_changed_during_preparation_fails_before_send(fresh_runner, monkeypatch):
    pilot, _, args, _ = fresh_runner
    original = pilot.video_images

    def changed(*values):
        result = original(*values)
        args['bundle'].write_bytes(b'changed trajectory')
        return result

    monkeypatch.setattr(pilot, 'video_images', changed)
    monkeypatch.setattr(pilot, 'call_vlm', lambda *_: pytest.fail('changed source was sent'))
    status = pilot.generate_once(**args)
    assert status['status'] == 'preparation_failed'
    assert 'Source changed' in status['error']
    assert status['vlm_calls'] == 0


def test_bimanual_parts_and_contact_predicates_still_required():
    plan, signals = example()
    plan['body_part_constraints'] = dict(mode='bimanual_upper_limbs_v1', basis='User-confirmed bimanual task')
    with pytest.raises(ValueError, match='both hand regions'):
        execute_event_program(plan, signals, 10)
    signals['right_hand_surface_distance'] = signals['left_hand_surface_distance'].copy()
    function = plan['actions'][0]['recognition']
    function['contact'] = dict(all=[function['contact'], dict(signal='right_hand_surface_distance', op='lt', value=0.05)])
    result = execute_event_program(plan, signals, 10)
    assert set(result['events'][0]['body_parts']) == {'left_hand', 'right_hand'}
    assert result['coverage']['key_frames'] == 10
    signals['right_hand_surface_distance'][:] = 0.2
    assert not execute_event_program(plan, signals, 10)['events']


@pytest.mark.parametrize('object_task', [True, False])
def test_prompt_template_has_structure_but_no_concrete_event_or_threshold(object_task):
    import json
    from holosoma_retargeting.semantic_keyframes.trajectory_events import event_prompt
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import ACTION_KEYS, FUNCTION_KEYS
    units = {'object_height': 'm'} if object_task else {'pelvis_speed': 'm/s'}
    prompt = event_prompt(units, bimanual=object_task)
    assert prompt.isascii()
    template, _ = json.JSONDecoder().raw_decode(prompt[prompt.index('{'):])
    action = template['actions'][0]
    assert set(action) == ACTION_KEYS and set(action['recognition']) == FUNCTION_KEYS
    assert action['action'] == '<unique_snake_case_event_name>'
    assert action['body_parts'] == ['<allowed_body_part>']
    assert action['recognition']['start']['value'] == '<finite_number>'
    assert 'example_transition' not in prompt
    assert '0.067' not in prompt and '0.04' not in prompt
    assert 'STRUCTURAL TEMPLATE ONLY' in prompt


def test_self_reflection_reviews_even_valid_initial_candidate(fresh_runner, monkeypatch):
    import json
    from pathlib import Path
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds'] = 2
    calls = []

    def vlm(images, prompt):
        calls.append((images, prompt))
        if len(calls) == 1:
            assert 'SELF-REFLECTION ON THIS RUN ONLY' not in prompt
            return json.dumps(plan)
        assert 'SELF-REFLECTION ON THIS RUN ONLY' in prompt
        assert 'per_event' in prompt and 'STRUCTURAL TEMPLATE ONLY' in prompt
        assert images == calls[0][0]
        return json.dumps(plan)

    monkeypatch.setattr(pilot, 'call_vlm', vlm)
    status = pilot.generate_once(**args)
    assert status['status'] == 'requires_visual_and_signal_review'
    assert status['vlm_calls'] == 2 and status['reflection_round'] == 1
    review = json.loads((Path(status['audit'])/'validation_1.json').read_text())
    assert review['valid'] and review['critique'] is None


def test_sr_can_fix_current_run_contract_without_reading_old_response(fresh_runner, monkeypatch):
    import json
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds'] = 2
    args['output_root'].mkdir()
    (args['output_root']/'response.txt').write_text('OLD_RUN_MUST_NOT_BE_READ')
    replies = ['CURRENT_RUN_INVALID_JSON', json.dumps(dict(critique='Corrected the current output structure.', program=plan))]
    calls = []

    def vlm(images, prompt):
        assert 'OLD_RUN_MUST_NOT_BE_READ' not in prompt
        if calls:
            assert 'CURRENT_RUN_INVALID_JSON' in prompt
        calls.append(prompt)
        return replies[len(calls)-1]

    monkeypatch.setattr(pilot, 'call_vlm', vlm)
    assert pilot.generate_once(**args)['status'] == 'requires_visual_and_signal_review'
    assert len(calls) == 2


def test_sr_is_bounded_and_never_accepts_invalid_last_round(fresh_runner, monkeypatch):
    from pathlib import Path
    pilot, _, args, _ = fresh_runner
    args['sr_rounds'] = 2
    calls = []
    monkeypatch.setattr(pilot, 'call_vlm', lambda *values: calls.append(values) or 'invalid')
    status = pilot.generate_once(**args)
    assert status['status'] == 'validation_failed' and len(calls) == 3
    assert not (Path(status['output'])/'semantic_plan.candidate.json').exists()


def test_quota_error_preserves_redacted_provider_detail_not_balance_claim(monkeypatch):
    import io
    import json
    import urllib.error
    from holosoma_retargeting.semantic_keyframes import pipeline
    secret = 'fake-test-key-never-log'
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://example.invalid/v1')
    monkeypatch.setenv('OPENAI_MODEL', 'test-model')
    monkeypatch.setenv('OPENAI_API_KEY', secret)
    body = json.dumps(dict(error=dict(code='insufficient_quota', type='quota_limit',
                                    message='Provider limit for ' + secret))).encode()

    def reject(*args, **kwargs):
        raise urllib.error.HTTPError('https://example.invalid', 429, 'quota', {}, io.BytesIO(body))

    monkeypatch.setattr(pipeline.urllib.request, 'urlopen', reject)
    with pytest.raises(pipeline.VLMQuotaError) as captured:
        pipeline.call_vlm([], 'local test')
    assert captured.value.provider_error['http_status'] == 429
    assert captured.value.provider_error['code'] == 'insufficient_quota'
    assert secret not in json.dumps(captured.value.provider_error)
    assert 'account balance not determined' in str(captured.value)


def test_staging_requires_sr_replay_and_unchanged_inputs(fresh_runner, monkeypatch, tmp_path):
    import json
    from pathlib import Path
    from stage_fresh_event_plans import stage
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds'] = 1
    calls = []

    def vlm(*values):
        calls.append(values)
        return json.dumps(plan if len(calls)==1 else dict(critique='Reviewed against the original images.',program=plan))

    monkeypatch.setattr(pilot,'call_vlm',vlm)
    row=dict(dataset='omomo',task='test_task',**pilot.generate_once(**args))
    result=stage(row,tmp_path/'reviewed','Test fixture reviewed')
    assert result['coverage']['ordinary_frames']==20
    published=json.loads((tmp_path/'reviewed/omomo/test_task/semantic_plan.json').read_text())
    assert published['generation_metadata']['approved_for_retargeting']
    (Path(row['output'])/'trajectory_signals.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Prepared input changed'):
        stage(row,tmp_path/'tampered','Must fail')
    row['status']='unresolved_functions'
    with pytest.raises(ValueError,match='Only valid nonempty candidates'):
        stage(row,tmp_path/'unresolved','Must fail')


def test_diagnostic_reports_exact_nested_paths_and_multiple_errors():
    import json
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
    plan, signals = example()
    bad = deepcopy(plan)
    action = bad['actions'][0]
    action['body_parts'] = ['left_leg']
    action['recognition']['start_hold_s'] = 0
    action['recognition']['contact']['op'] = 'gt'
    report = diagnose_response(json.dumps(dict(critique='review',program=bad)), {}, signals, 10, reflected=True)
    paths={item['path']:item for item in report['errors']}
    assert '$.program.actions[0].body_parts[0]' in paths
    assert paths['$.program.actions[0].body_parts[0]']['actual']=='left_leg'
    assert '$.program.actions[0].recognition.start_hold_s' in paths
    assert paths['$.program.actions[0].recognition.contact.op']['actual']=='gt'
    assert all(item['event']=='lift' for item in report['errors'])


def test_diagnostic_duplicate_reports_both_function_locations():
    import json
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
    plan, signals=example()
    duplicate=deepcopy(plan['actions'][0]);duplicate['action']='different_event'
    plan['actions'].append(duplicate)
    report=diagnose_response(json.dumps(plan),{},signals,10)
    conflict=next(item for item in report['errors'] if 'conflicts_with' in item)
    assert conflict['path']=='$.actions[1].recognition'
    assert conflict['conflicts_with']=='$.actions[0].recognition'


def test_diagnostic_no_contact_shows_actual_gate_and_signal_range():
    import json
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
    plan, signals=example();signals['left_hand_surface_distance'][:]=0.2
    report=diagnose_response(json.dumps(plan),{},signals,10)
    row=report['per_event'][0]
    assert row['gate_counts']['start']>0 and row['gate_counts']['contact']==0
    assert row['gate_counts']['joint_eligible']==0
    leaf=next(item for item in row['predicate_checks'] if item['path'].endswith('.contact'))
    assert leaf['signal_min']==0.2 and leaf['threshold']==0.05 and leaf['true_frames']==0
    assert any(item['path']=='$.actions[0].recognition.contact' for item in report['warnings'])


def test_diagnostic_json_syntax_location():
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
    _, signals=example()
    report=diagnose_response('{"schema":\n invalid}',{},signals,10)
    error=report['errors'][0]
    assert error['path']=='$' and error['json_block_line']==2
    assert error['json_block_column']>0


def test_six_sr_rounds_deliver_exact_errors_and_stop_at_seven_calls(fresh_runner, monkeypatch):
    import json
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds']=6
    bad=deepcopy(plan);bad['actions'][0]['body_parts']=['left_leg']
    calls=[]
    def vlm(images,prompt):
        calls.append(prompt)
        if len(calls)>1:
            assert 'body_parts[0]' in prompt and 'left_leg' in prompt
        # Distinct failures exercise the six-round ceiling, not the stagnation guard.
        bad['actions'][0]['body_parts'] = ['left_leg_' + str(len(calls))]
        return json.dumps(bad)
    monkeypatch.setattr(pilot,'call_vlm',vlm)
    status=pilot.generate_once(**args)
    assert status['vlm_calls']==7 and status['reflection_round']==6
    assert status['status']=='validation_failed'


def test_sr_sends_only_previous_plan_and_matching_paths_without_old_critique():
    import json
    from pilot_trajectory_event_plans import reflection_prompt
    plan, _ = example()
    raw = json.dumps(dict(critique='STALE_CRITIQUE_DO_NOT_SEND', program=plan))
    feedback = dict(errors=[dict(path='$.program.actions[0].recognition.start',
                                reason='CURRENT_ERROR', actual=plan['actions'][0]['recognition']['start'],
                                conflicts_with='$.program.actions[1].recognition.start')])
    prompt = reflection_prompt('STRUCTURAL_TEMPLATE', raw, feedback, round_index=3, unchanged=True)
    assert 'STALE_CRITIQUE_DO_NOT_SEND' not in prompt
    assert 'CURRENT_ERROR' in prompt and '$.actions[0].recognition.start' in prompt
    assert '$.actions[1].recognition.start' in prompt and '$.program.actions' not in prompt
    assert json.dumps(plan, indent=2) in prompt
    assert 'Round 3' in prompt and 'left the program unchanged' in prompt


def test_sr_always_uses_immediately_previous_plan_and_error(fresh_runner, monkeypatch):
    import json
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds'] = 2
    first, second = deepcopy(plan), deepcopy(plan)
    first['actions'][0]['body_parts'] = ['FIRST_INVALID_PART']
    second['actions'][0]['body_parts'] = ['SECOND_INVALID_PART']
    replies = [first, second, plan]
    calls = []
    def vlm(images, prompt):
        if len(calls) == 1:
            assert 'FIRST_INVALID_PART' in prompt and 'body_parts[0]' in prompt
        if len(calls) == 2:
            assert 'SECOND_INVALID_PART' in prompt and 'body_parts[0]' in prompt
            assert 'FIRST_INVALID_PART' not in prompt
        calls.append(prompt)
        return json.dumps(replies[len(calls)-1])
    monkeypatch.setattr(pilot, 'call_vlm', vlm)
    status = pilot.generate_once(**args)
    assert status['status'] == 'requires_visual_and_signal_review' and len(calls) == 3


def test_initial_and_sr_use_same_validation_contract():
    import json
    from pilot_trajectory_event_plans import resolve_candidate
    from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
    plan, signals = example()
    for reflected in (False, True):
        for raw in (json.dumps(plan), json.dumps(dict(critique='legacy transport', program=plan))):
            parsed, result, _ = resolve_candidate(raw, {}, signals, 10, reflected=reflected)
            assert parsed == plan and result['events']
            assert not diagnose_response(raw, {}, signals, 10, reflected=reflected)['errors']


def test_sr_stops_identical_failures_without_accepting_or_spending_all_rounds(fresh_runner, monkeypatch):
    import json
    from pathlib import Path
    pilot, plan, args, _ = fresh_runner
    args['sr_rounds'] = 6
    plan['actions'][0]['body_parts'] = ['unknown_part']
    calls = []
    def vlm(images, prompt):
        calls.append(prompt)
        return json.dumps(dict(critique='Variable irrelevant critique ' + str(len(calls)), program=plan))
    monkeypatch.setattr(pilot, 'call_vlm', vlm)
    status = pilot.generate_once(**args)
    assert status['status'] == 'validation_failed'
    assert status['vlm_calls'] == len(calls) == 3
    assert status['stop_reason'] == 'three_identical_programs_and_diagnostics'
    assert 'left the program unchanged' in calls[-1]
    assert not (Path(status['output'])/'semantic_plan.candidate.json').exists()
