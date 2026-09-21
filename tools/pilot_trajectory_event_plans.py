#!/usr/bin/env python3
"""Fresh whole-clip event generation, followed by trajectory-only localization.

Each invocation prepares new inputs, then reflects only on its own new candidate.
No historical response, event plan, or visual phase cache is read.
Candidates remain review-only; this tool never activates training inputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np

from holosoma_retargeting.semantic_keyframes.local_phases import video_images, skeleton_images
from holosoma_retargeting.semantic_keyframes.pipeline import load_env_file, call_vlm
from holosoma_retargeting.semantic_keyframes.event_diagnostics import diagnose_response
from holosoma_retargeting.semantic_keyframes.trajectory_events import (
    SCHEMA, load_event_signals, event_prompt, execute_event_program, unpack_event_response,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'src/holosoma_retargeting/holosoma_retargeting/demo_data'
VERSION = 'trajectory_events_fresh_v1_20260920'
AUTHORIZED_HOST = 'llm-1cbh3bu15hok8ksy.cn-beijing.maas.aliyuncs.com'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve_candidate(raw, constraints, signals, fps, *, reflected=False):
    parsed, critique, _ = unpack_event_response(raw)
    if not isinstance(parsed, dict) or set(parsed) != {'schema', 'actions'}:
        raise ValueError('Expected exactly schema and actions; no visual timing or phase fields')
    if any(action.get('action') == 'example_transition' for action in parsed.get('actions', []) if isinstance(action, dict)):
        raise ValueError('The syntax example is not a video-grounded event annotation')
    if constraints:
        parsed['body_part_constraints'] = constraints
    result = execute_event_program(parsed, signals, fps)
    return parsed, result, critique


def reflection_prompt(original_prompt, raw, feedback, *, round_index=1, max_rounds=6,
                      unchanged=False):
    """Review only the immediately preceding program with matching, compact evidence."""
    try:
        program, _, base = unpack_event_response(raw)
        candidate = json.dumps(program, ensure_ascii=False, indent=2)
        label = 'Immediately previous semantic plan from THIS RUN'
    except (ValueError, TypeError):
        # No parsed plan exists. Preserve the malformed response and its parse location.
        candidate, base = raw, '$'
        label = 'Immediately previous malformed response (no parseable semantic plan)'

    def paths(value):
        if isinstance(value, dict):
            return {key: (item.replace('$.program', '$', 1)
                          if key in ('path', 'conflicts_with') and isinstance(item, str)
                          and base == '$.program' else paths(item)) for key, item in value.items()}
        if isinstance(value, list):
            return [paths(item) for item in value]
        return value

    # Do not repeat recognition functions in both full execution output and per-event
    # diagnostics. All issue locations/reasons survive, including syntax line/column.
    diagnostic = {key: feedback[key] for key in
                  ('error', 'errors', 'warnings', 'coverage', 'unresolved_events') if key in feedback}
    evidence = []
    for item in feedback.get('per_event', []):
        if item.get('valid') and not item.get('unresolved'):
            evidence.append({key: item[key] for key in ('path', 'event', 'valid', 'coverage', 'intervals') if key in item})
        else:
            evidence.append(item)
    diagnostic['per_event'] = evidence
    return (
        original_prompt + '\n\nSELF-REFLECTION ON THIS RUN ONLY. '
        f'Round {round_index} of at most {max_rounds}. '
        'The attached images are the same original whole-clip inputs. Reinspect them. '
        'Use the immediately previous semantic plan below and its local validation feedback. '
        'There are no historical-run plans or earlier critiques in this request. '
        'First correct schema errors at the supplied JSON paths. Then inspect event meaning, '
        'signal units, contact gates, onset, termination and duration using the supplied evidence. '
        'Warnings are diagnostic hypotheses to assess against the images, not invented target intervals. '
        'Retain supported events and correct unsupported claims; do not drop a visible critical event '
        'merely to pass validation. Do not widen thresholds merely to force a match. '
        'If the required evidence is unavailable, explain that limitation in function_rationale. '
        + ('The immediately previous attempt left the program unchanged. The listed problems remain; '
           're-evaluate their specific causes instead of repeating the same output. ' if unchanged else '')
        + 'Return the COMPLETE revised program in EXACTLY the SAME schema/actions format as the initial '
        'request. No critique/program wrapper, no patch, no additional top-level fields. '
        'In function_rationale briefly state why the revised conditions recognize the event.\n'
        + label + ':\n' + candidate + '\n'
        'Errors, locations, reasons and trajectory evidence for that plan:\n'
        + json.dumps(paths(diagnostic), ensure_ascii=False))


def generate_once(*, bundle, video, constraints, output_root, audit_root, samples=40,
                  prepare_only=False, lafan_source=None, sr_rounds=6):
    """Generate fresh, then optionally self-reflect within this run, never on old files."""
    if samples < 2:
        raise ValueError('At least two chronological whole-clip samples are required')
    if type(sr_rounds) is not int or not 0 <= sr_rounds <= 6:
        raise ValueError('sr_rounds must be an integer between 0 and 6')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_' + uuid4().hex[:8]
    output, audit = output_root / 'runs' / run_id, audit_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    audit.mkdir(parents=True, exist_ok=False)
    status = dict(run_id=run_id, status='preparing', review_only=True,
                  active_for_retargeting=False, output=str(output), audit=str(audit), vlm_calls=0)
    write_json(audit / 'status.json', status)
    stage = 'preparation'
    try:
        if lafan_source is not None:
            from holosoma_retargeting.config_types.data_type import LAFAN_DEMO_JOINTS
            source_hashes = {str(lafan_source): file_hash(lafan_source)}
            joints = np.load(lafan_source, allow_pickle=False)[:, :, [0, 2, 1]]
            bundle = output / 'lafan_gt_sequence.npz'
            np.savez_compressed(bundle, human_joints=joints, smplh_joint_names=np.asarray(LAFAN_DEMO_JOINTS),
                                human_joint_layout=np.asarray('lafan_z_up_v1'),
                                frame_ids=np.arange(len(joints)), fps=np.asarray(30.0),
                                source_file=np.asarray(str(lafan_source)))
        else:
            source_hashes = {str(path): file_hash(path) for path in (bundle, video)}
        signals, units, fps = load_event_signals(bundle)
        exposed = {'object_height', 'object_speed', 'object_vertical_velocity', 'object_angular_speed',
                   'object_rotate',
                   'pelvis_speed', 'pelvis_height', 'left_hand_speed', 'right_hand_speed'}
        exposed.update(k for k in signals if k.endswith('_surface_distance'))
        if lafan_source is not None:
            exposed.update(signals)
        signals = {key: value for key, value in signals.items() if key in exposed}
        units = {key: units[key] for key in signals}
        write_json(output / 'signal_units.json', units)
        np.savez_compressed(output / 'trajectory_signals.npz', fps=fps, **signals)
        count = len(next(iter(signals.values())))
        frames = np.unique(np.linspace(0, count - 1, min(samples, count)).round().astype(int))
        # New run directory prevents reuse of any previous image/phase cache.
        images = (skeleton_images(joints, output / 'whole_clip_visual_inputs', lafan_source.stem, count=len(frames))
                  if lafan_source is not None else
                  video_images(video, frames, fps, output / 'whole_clip_visual_inputs'))
        if not images:
            raise ValueError('A fresh visual input is required for event generation')
        prompt = event_prompt(units, bimanual=constraints.get('mode') == 'bimanual_upper_limbs_v1')
        if lafan_source is not None:
            prompt += ('\nThis is a robot-only LAFAN motion, with no interaction object. The chronological '
                       'skeleton sheets show the whole sequence in X/Z and Y/Z views (left red, right blue). '
                       'Use only the available body height/speed signals. For body-only events use '
                       'contact=null, contact_timing="none", contact_lookback_s=0. Do not invent object signals.')
        ranges = {key: dict(min=float(np.min(value)), max=float(np.max(value)))
                  for key, value in signals.items()}
        prompt += '\nTrajectory signal ranges for physical threshold calibration (not timing): ' + json.dumps(ranges)
        (audit / 'prompt.txt').write_text(prompt)
        write_json(output / 'body_part_constraints.json', constraints)
        # Same request body as call_vlm; credentials are never saved.
        request = dict(model=os.environ.get('OPENAI_MODEL'), temperature=0, seed=0,
                       messages=[dict(role='user', content=[dict(type='text', text=prompt), *images])])
        write_json(output / 'request.json', request)
        package = ROOT / 'src/holosoma_retargeting/holosoma_retargeting/semantic_keyframes'
        code_files = [Path(__file__).resolve(), *(package / name for name in
                      ('trajectory_events.py', 'event_diagnostics.py', 'pipeline.py', 'local_phases.py', 'body_constraints.py'))]
        code_hashes = {}
        for path in code_files:
            relative = path.relative_to(ROOT)
            target = audit / 'code_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            code_hashes[str(relative)] = file_hash(target)
        if any(file_hash(Path(path)) != digest for path, digest in source_hashes.items()):
            raise ValueError('Source changed during input preparation; discard this run')
        endpoint = urlsplit(os.environ.get('OPENAI_BASE_URL', ''))
        manifest = dict(schema=SCHEMA, workflow='whole_clip_events_then_trajectory_predicates',
                        generation_policy=('fresh_run_current_run_self_reflection' if sr_rounds else
                                           'fresh_single_request_no_history_no_feedback'),
                        max_self_reflection_rounds=sr_rounds,
                        run_id=run_id, source_sha256=source_hashes, code_sha256=code_hashes,
                        input_sha256={str(path.relative_to(output)): file_hash(path)
                                      for path in sorted(output.rglob('*')) if path.is_file()},
                        prompt_sha256=file_hash(audit / 'prompt.txt'), fps=fps, trajectory_frames=count,
                        sample_frames=frames.tolist(), image_count=len(images),
                        visual_input='uniform chronological samples spanning the entire clip in one context',
                        model=request['model'], temperature=0, seed=0,
                        endpoint_host=endpoint.hostname, endpoint_path=endpoint.path,
                        python=sys.version, numpy=np.__version__,
                        body_part_constraints=constraints,
                        review_only=True, active_for_retargeting=False,
                        reproducibility='Exact request and code recorded; remote model determinism is not guaranteed. '
                                        'The saved program and signals support deterministic local replay.')
        write_json(audit / 'manifest.json', manifest)
        if prepare_only:
            status['status'] = 'prepared_only'
        else:
            if endpoint.scheme != 'https' or endpoint.hostname != AUTHORIZED_HOST or endpoint.username or endpoint.password:
                raise ValueError('Unexpected VLM endpoint; refusing to send to a different service')
            if not request['model'] or not os.environ.get('OPENAI_API_KEY'):
                raise ValueError('VLM model and credentials must be configured')
            request_prompt = prompt
            previous_program = None
            previous_fingerprint, identical_attempts = None, 0
            for round_index in range(sr_rounds + 1):
                stage = 'request'
                status['vlm_calls'] += 1
                status['status'] = 'requesting'
                status['reflection_round'] = round_index
                write_json(audit / 'status.json', status)
                (audit / f'prompt_{round_index}.txt').write_text(request_prompt)
                round_request = dict(request, messages=[dict(role='user', content=[
                    dict(type='text', text=request_prompt), *images])])
                write_json(output / f'request_{round_index}.json', round_request)
                raw = call_vlm(images, request_prompt)
                (audit / f'response_{round_index}.txt').write_text(raw)
                if round_index == 0:
                    (audit / 'response.txt').write_text(raw)
                stage = 'validation'
                diagnostics = diagnose_response(raw, constraints, signals, fps, reflected=round_index > 0)
                try:
                    current_program = json.dumps(unpack_event_response(raw)[0], sort_keys=True)
                except (ValueError, TypeError):
                    current_program = raw
                fingerprint = json.dumps([current_program, diagnostics], sort_keys=True)
                identical_attempts = identical_attempts + 1 if fingerprint == previous_fingerprint else 1
                previous_fingerprint = fingerprint
                stagnant = identical_attempts >= 3
                if stagnant:
                    status['stop_reason'] = 'three_identical_programs_and_diagnostics'
                try:
                    plan, result, critique = resolve_candidate(raw, constraints, signals, fps, reflected=round_index > 0)
                    feedback = dict(coverage=result['coverage'], unresolved_events=result['unresolved_events'],
                                    recognition_diagnostics=result['recognition_diagnostics'],
                                    **diagnostics)
                    write_json(audit / f'validation_{round_index}.json', dict(valid=True, critique=critique, **feedback))
                    # Always perform at least one SR pass when enabled, even if the first candidate validates.
                    if stagnant or not sr_rounds or (round_index > 0 and not result['unresolved_events']
                                         and not diagnostics['errors'] and not diagnostics['warnings']):
                        break
                except (ValueError, TypeError, KeyError) as exc:
                    feedback = dict(valid=False, error=str(exc), **diagnostics)
                    write_json(audit / f'validation_{round_index}.json', feedback)
                    if stagnant or round_index == sr_rounds:
                        raise
                if round_index < sr_rounds:
                    request_prompt = reflection_prompt(
                        prompt, raw, feedback, round_index=round_index + 1, max_rounds=sr_rounds,
                        unchanged=current_program == previous_program)
                    previous_program = current_program
            result['candidate_status'] = ('no_event_proposal' if not plan['actions'] else
                                          'unresolved_functions' if result['unresolved_events'] else
                                          'semantic_review_required' if diagnostics['errors'] or diagnostics['warnings'] else
                                          'requires_visual_and_signal_review')
            result['generation_metadata'] = dict(run_id=run_id, manifest=str(audit / 'manifest.json'),
                                                review_only=True, active_for_retargeting=False,
                                                generation_policy=manifest['generation_policy'],
                                                self_reflection_rounds=round_index,
                                                body_part_constraints=constraints)
            write_json(output / 'event_program.json', plan)
            write_json(output / 'semantic_plan.candidate.json', result)
            write_json(audit / 'resolution.json', result)
            status.update(status=result['candidate_status'], coverage=result['coverage'])
    except Exception as exc:
        status.update(status=stage + '_failed', error_type=type(exc).__name__)
        if getattr(exc, 'provider_error', None):
            status['provider_error'] = exc.provider_error
        # Provider errors can echo request data; retain only the exception type there.
        if stage != 'request':
            status['error'] = str(exc)
    write_json(audit / 'status.json', status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tasks', nargs='+', default=['sub1_largetable_028', 'sub16_largetable_013'])
    parser.add_argument('--samples', type=int, default=40)
    parser.add_argument('--sr-rounds', type=int, choices=range(7), default=6)
    parser.add_argument('--prepare-only', action='store_true', help='Prepare fresh inputs without any network request')
    args = parser.parse_args()
    if not args.prepare_only:
        load_env_file(ROOT / 'src/holosoma_retargeting/.env')
    rules = json.loads((DATA / 'semantic_keyframes/g1_anatomy_v2_20260920/task_body_constraints.json').read_text())
    results = []
    for task in args.tasks:
        if 'omomo/' + task not in rules:
            parser.error(f'Unknown unfinished OMOMO task: {task}')
        result = generate_once(
            bundle=DATA / 'omomo/bundles' / task / 'input/omomo_gt_sequence.npz',
            video=DATA / 'omomo/bundles' / task / 'videos' / f'{task}_rerender.mp4',
            constraints=rules['omomo/' + task],
            output_root=DATA / 'semantic_keyframes' / VERSION / 'omomo' / task,
            audit_root=ROOT / 'exp/semantic_plans' / VERSION / 'omomo' / task,
            samples=args.samples, prepare_only=args.prepare_only, sr_rounds=args.sr_rounds)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(any(item['status'] not in ('prepared_only', 'requires_visual_and_signal_review') for item in results))


if __name__ == '__main__':
    raise SystemExit(main())
