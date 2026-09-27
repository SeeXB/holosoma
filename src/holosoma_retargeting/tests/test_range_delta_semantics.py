"""Signed amplitude ratios: direction, references, zero values and failure audit."""
import json

import numpy as np
import pytest

from holosoma_retargeting.semantic_keyframes import pipeline as p
from holosoma_retargeting.semantic_keyframes import range_delta as r


def rule(ratio=.5, crossing="up", times=(0,6), reference="clip_start", signal="object_height"):
    return dict(signal=signal, ratio=ratio, crossing=crossing, reference=reference, time_range_s=list(times))


def signals(values=(0,1,2,3,2,1,0)):
    return {"object_height":np.asarray(values,dtype=float)}


def plan(start=None, end=None):
    return dict(schema=r.SCHEMA, actions=[dict(action="raise_and_lower", body_parts=["left_hand"],
        criticality_level=3, rationale="Visible raising then lowering", start=start or rule(times=(0,3)),
        end=end or rule(crossing="down",times=(3,6)))])


def test_zero_initial_value_and_same_positive_ratio_crossed_in_opposite_directions():
    up=r.resolve_boundary(rule(),signals(),1)
    down=r.resolve_boundary(rule(crossing="down"),signals(),1)
    assert up['frame']==2 and down['frame']==5
    assert up['reference_value']==0 and up['signal_span']==3
    assert up['ratio_at_frame']>0 and down['ratio_at_frame']>0


def test_negative_ratio_means_below_reference_not_arbitrary_downward_motion():
    bad=r.resolve_boundary(rule(-.5,"down"),signals(),1)
    good=r.resolve_boundary(rule(-.5,"down",(3,6),"range_start"),signals(),1)
    assert bad['status']=='unresolved' and bad['frame'] is None
    assert good['frame']==5 and good['reference_frame']==3


def test_whole_clip_amplitude_not_window_amplitude_or_quantile():
    data=signals((0,0,0,0,2,4,8,10,10))
    result=r.resolve_boundary(rule(.3,times=(3,5)),data,1)
    assert result['signal_span']==10
    assert result['raw_threshold']==3
    assert result['frame']==5


def test_range_start_reference_changes_when_visual_window_changes():
    a=r.resolve_boundary(rule(.5,'down',(3,6),'range_start'),signals(),1)
    b=r.resolve_boundary(rule(-.5,'down',(4,6),'range_start'),signals(),1)
    assert a['frame'] is None
    assert b['reference_value']==2 and b['frame']==6


def test_action_start_reference_is_resolved_frame_and_not_window_beginning():
    payload=plan(end=rule(-.5,'down',(3,6),'action_start'))
    result=r.resolve_plan(payload,signals(),1)
    end=result['actions'][0]['boundaries']['end']
    assert result['status']=='resolved'
    assert end['reference_frame']==2 and end['reference_value']==2 and end['frame']==6
    replay=p.execute_dynamic_plan(payload,signals(),1)
    assert replay['events']==result['resolved_events']


def test_translation_and_positive_unit_scaling_do_not_change_ratio_crossings():
    raw=signals()['object_height']
    first=r.resolve_boundary(rule(),signals(raw),1)
    transformed=r.resolve_boundary(rule(),signals(raw*1000+2345),1)
    assert first['frame']==transformed['frame']
    assert first['ratio_before']==transformed['ratio_before']
    assert first['ratio_at_frame']==transformed['ratio_at_frame']


def test_constant_signal_unresolved_without_epsilon_fallback():
    result=r.resolve_boundary(rule(),signals(np.ones(7)),1)
    assert result['frame'] is None and result['signal_span']==0
    assert 'zero' in result['reason']


def test_small_noise_is_not_falsely_claimed_as_filtered_physical_motion():
    # Deliberately document the tradeoff: scale invariance also magnifies jitter.
    result=r.resolve_boundary(rule(),signals(np.array([0,1,2,3,2,1,0])*1e-6),1)
    assert result['frame']==2
    assert result['signal_span']==pytest.approx(3e-6)


def test_does_not_escape_search_range_or_use_nearest_value():
    result=r.resolve_boundary(rule(.5,times=(3,4)),signals(),1)
    assert result['frame'] is None and not result['candidate_frames']


def test_already_true_condition_is_not_a_crossing_inside_range():
    result=r.resolve_boundary(rule(.1,times=(2,3)),signals(),1)
    assert result['frame'] is None


def test_zero_ratio_can_detect_return_below_reference():
    result=r.resolve_boundary(rule(0,'down'),signals((1,2,3,2,1,0,-1)),1)
    assert result['frame']==4


def test_multiple_real_matches_are_retained_and_earliest_selected():
    result=r.resolve_boundary(rule(.5),signals((0,2,0,2,0,2,0)),1)
    assert result['candidate_frames']==[1,3,5]
    assert result['frame']==1 and result['multiple_candidates']


def test_all_schema_errors_reported_together_and_circular_references_rejected():
    payload=plan()
    action=payload['actions'][0]
    action['body_parts']=['object']
    action['start']['reference']='action_start'
    action['end']['ratio']=float('nan')
    action['end']['signal']='invented_signal'
    issues=r.validation_issues(payload,{'object_height'},6)
    assert len(issues)==4
    with pytest.raises(ValueError):r.resolve_plan(payload,signals(),1)


@pytest.mark.parametrize('bad',[
    rule(ratio=True),rule(ratio='20%'),rule(ratio=2),rule(times=(-1,2)),
    dict(rule(),q=.5),rule(crossing='global_minimum'),
    rule(-.2,'down',signal='object_rotate'),
])
def test_old_grammar_and_invalid_values_rejected(bad):
    assert r.validation_issues(plan(start=bad),{'object_height','object_rotate'},6)


def test_edge_strings_are_censored_and_no_zero_duration_export():
    result=r.resolve_plan(plan('clip_start','clip_end'),signals(),1)
    assert result['status']=='resolved'
    assert all(x['status']=='clip_censored' for x in result['actions'][0]['boundaries'].values())
    bad=plan(end=rule(.5,'up',times=(2,6)))
    result=r.resolve_plan(bad,signals(),1)
    assert result['status']=='unresolved'
    with pytest.raises(ValueError):p.execute_dynamic_plan(bad,signals(),1)


def test_failed_start_reference_does_not_create_partial_accepted_plan():
    payload=plan(rule(-.2,'down',times=(0,3)),rule(.2,'up',(3,6),'action_start'))
    result=r.resolve_plan(payload,signals(),1)
    assert result['status']=='unresolved'
    assert result['actions'][0]['boundaries']['end']['reason']=='action-start reference unresolved'
    assert not result['resolved_events']


def test_full_video_generation_repairs_all_fields_and_keeps_raw_candidates(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4';video.write_bytes(b'test MP4')
    monkeypatch.setattr(p,'load_retargeting_bundle_signals',lambda _: (signals(),{},1))
    monkeypatch.setenv('OPENAI_MODEL','mock')
    calls=[]
    bad=plan();bad['actions'][0]['body_parts']=['object'];bad['actions'][0]['start']['reference']='action_start'
    def call(content,prompt):
        calls.append((content,prompt))
        return json.dumps(bad if len(calls)==1 else plan())
    output=tmp_path/'semantic_plan.json'
    result=r.generate_range_delta_plan(video=video,bundle_file=video,output=output,max_repairs=1,call=call)
    assert calls[0][0][0]['type']=='video_url'
    assert 'invalid or duplicate body regions' in calls[1][1] and 'circular' in calls[1][1]
    assert result['generation_metadata']['vlm_attempt_count']==2
    assert (tmp_path/'audit/attempt_0.response.txt').exists()


def test_unresolved_generation_keeps_diagnostics_without_export(tmp_path,monkeypatch):
    video=tmp_path/'video.mp4';video.write_bytes(b'test')
    monkeypatch.setattr(p,'load_retargeting_bundle_signals',lambda _: (signals(),{},1))
    output=tmp_path/'semantic_plan.json'
    with pytest.raises(ValueError,match='whole plan remains unresolved'):
        r.generate_range_delta_plan(video=video,bundle_file=video,output=output,max_repairs=0,
                                     call=lambda *_:json.dumps(plan(rule(-.5,'down',times=(0,3)))))
    assert not output.exists() and (tmp_path/'audit/attempt_0.resolution.json').exists()
