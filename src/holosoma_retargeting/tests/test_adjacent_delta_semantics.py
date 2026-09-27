"""Adjacent-frame math, one numeric output, automatic runs and chronology."""
import json

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes import adjacent_delta as a
from holosoma_retargeting.semantic_keyframes import pipeline as p


def action(condition="increasing", threshold=.2, signal="object_height", name="move"):
    return dict(action=name,body_parts=["left_hand"],signal=signal,condition=condition,
                threshold=threshold,rationale="Visible action supports this signal")


def plan(*actions):
    return dict(schema=a.SCHEMA,actions=list(actions))


def signals(values=(0,0,1,2,3,3,3,2,1,0,0)):
    return {"object_height":np.asarray(values,dtype=float)}


def test_delta_is_previous_to_current_not_chosen_reference_or_cumulative():
    ratio,span=a.delta_ratios([0,1,2,3,2,1,0])
    assert span==3 and np.isnan(ratio[0])
    np.testing.assert_allclose(ratio[1:],[1/3,1/3,1/3,-1/3,-1/3,-1/3])


def test_sign_directly_distinguishes_rise_fall_without_crossing_field():
    data=signals()
    up=a.resolve_action(action(),data)
    down=a.resolve_action(action("decreasing"),data)
    assert up['window']==[1,4] and up['trigger_frame']==2
    assert down['window']==[6,9] and down['trigger_frame']==7
    assert up['first_ratio']>0 and down['first_ratio']<0


def test_start_hold_lower_are_located_by_order_with_one_threshold_each():
    payload=plan(action(),action("steady",.01,name="hold"),action("decreasing",name="lower"))
    result=a.resolve_plan(payload,signals(),30)
    assert result['status']=='resolved'
    assert [e['windows'][0] for e in result['resolved_events']]==[
        dict(start_frame=1,end_frame=4,trigger_frame=2),
        dict(start_frame=4,end_frame=6,trigger_frame=5),
        dict(start_frame=6,end_frame=9,trigger_frame=7),
    ]
    assert p.execute_dynamic_plan(payload,signals(),30)['events']==result['resolved_events']


def test_repeated_actions_match_distinct_complete_runs():
    data=signals((0,1,2,2,1,0,0,1,2,2))
    result=a.resolve_plan(plan(action(name="rise1"),action(name="rise2")),data,30)
    assert [t['selected_run'] for t in result['actions']]==[[1,2],[7,8]]


def test_does_not_split_continuing_run_to_invent_repetition():
    result=a.resolve_plan(plan(action(name="rise1"),action(name="rise2")),signals(),30)
    assert result['status']=='unresolved'
    assert result['actions'][1]['window'] is None


def test_single_step_is_retained_flagged_and_never_zero_duration():
    trace=a.resolve_action(action(),signals((0,0,3,3,3)))
    assert trace['window']==[1,2] and trace['matching_steps']==1
    assert any('single adjacent step' in s for s in trace['warnings'])


def test_unknown_first_delta_never_manufactures_activity_at_frame_zero():
    trace=a.resolve_action(action("steady",0),signals((0,1,1,0)))
    assert trace['selected_run']==[2,2] and trace['window']==[1,2]


def test_constant_signal_undefined_and_tiny_amplitude_not_silently_filtered():
    bad=a.resolve_action(action("steady",.1),signals((2,2,2)))
    assert bad['window'] is None and bad['signal_span']==0
    small=a.resolve_action(action(),signals(np.array([0,0,1,2,3,3])*1e-9))
    assert small['window']==[1,4]


def test_scale_translation_invariance_and_native_per_frame_units():
    data=signals()['object_height']
    first=a.resolve_action(action(),signals(data))
    second=a.resolve_action(action(),signals(data*1000+17))
    assert first['window']==second['window']
    assert a.resolve_plan(plan(action()),signals(),30)['actions']==a.resolve_plan(plan(action()),signals(),60)['actions']


def test_whole_clip_amplitude_not_difference_range():
    ratios,span=a.delta_ratios([0,1,2,2,3,3])
    assert span==3
    assert ratios[1]==pytest.approx(1/3)


def test_no_match_does_not_fall_back_and_later_actions_are_blocked():
    payload=plan(action(threshold=.9),action("decreasing",name="lower"))
    result=a.resolve_plan(payload,signals(),30)
    assert result['status']=='unresolved' and not result['resolved_events']
    assert result['actions'][0]['status']=='unresolved'
    assert result['actions'][1]['status']=='blocked_by_previous_action'
    assert result['actions'][1]['window'] is None
    assert result['actions'][1]['candidate_runs']
    assert 'selected_run' not in result['actions'][1]
    with pytest.raises(ValueError):p.execute_dynamic_plan(payload,signals(),30)


@pytest.mark.parametrize('extra',[
    {'reference':'clip_start'},{'time_range_s':[0,1]},{'start':0},{'end':1},
    {'crossing':'up'},{'criticality_level':3},{'frame':17},{'lag':1},
])
def test_vlm_time_reference_and_additional_numeric_fields_forbidden(extra):
    assert a.validation_issues(plan(dict(action(),**extra)),{'object_height'})


@pytest.mark.parametrize('threshold',[True,-.1,1.1,float('nan'),float('inf'),'1%'])
def test_bad_threshold_rejected(threshold):
    assert a.validation_issues(plan(action(threshold=threshold)),{'object_height'})


def test_zero_threshold_only_for_steady_and_cumulative_signal_cannot_decrease():
    assert a.validation_issues(plan(action(threshold=0)),{'object_height'})
    assert not a.validation_issues(plan(action('steady',0)),{'object_height'})
    assert a.validation_issues(plan(action('decreasing',signal='body_travel_progress')),{'body_travel_progress'})


def test_whole_clip_condition_flagged_and_edges_censored_automatically():
    result=a.resolve_plan(plan(action('steady',1)),signals(),30)
    trace=result['actions'][0]
    assert trace['left_clip_censored'] and trace['right_clip_censored']
    assert any('no internal segmentation' in x for x in trace['warnings'])
    event=result['resolved_events'][0]
    assert all(v['status']=='clip_censored' for v in event['boundary_evidence'].values())


def test_full_video_generation_keeps_exact_single_quantity_and_repairs(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4';video.write_bytes(b'test')
    monkeypatch.setattr(p,'load_retargeting_bundle_signals',lambda _:(signals(),{},30))
    monkeypatch.setenv('OPENAI_MODEL','mock')
    calls=[]
    def call(content,prompt):
        calls.append((content,prompt))
        return json.dumps(plan(dict(action(),time_range_s=[0,1])) if len(calls)==1 else plan(action()))
    result=a.generate_adjacent_delta_plan(video=video,bundle_file=video,output=tmp_path/'out.json',max_repairs=1,call=call)
    assert calls[0][0][0]['type']=='video_url'
    assert result['generation_metadata']['numeric_parameters_per_action']==1
    assert not result['generation_metadata']['vlm_selects_time_or_reference']
    saved=json.loads((tmp_path/'out.event_plan.json').read_text())
    assert [k for k,v in saved['actions'][0].items() if type(v) in (int,float)]==['threshold']
    assert (tmp_path/'audit/attempt_0.validation.json').exists()


def test_failed_generation_does_not_export_partial_success(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4';video.write_bytes(b'test')
    monkeypatch.setattr(p,'load_retargeting_bundle_signals',lambda _:(signals(),{},30))
    output=tmp_path/'out.json'
    with pytest.raises(ValueError,match='whole plan remains unresolved'):
        a.generate_adjacent_delta_plan(video=video,bundle_file=video,output=output,max_repairs=0,
            call=lambda *_:json.dumps(plan(action(),action(name='unsupported repeat'))))
    assert not output.exists() and (tmp_path/'audit/attempt_0.resolution.json').exists()
