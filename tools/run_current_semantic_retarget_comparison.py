#!/usr/bin/env python3
"""Paired Original/B4 retargeting and method-independent evaluation of current plans."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import numpy as np

from prepare_batch_retarget_inputs import TASK_OBJECTS
from prepare_lafan_batch_inputs import TASKS as LAFAN_TASKS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/holosoma_retargeting/holosoma_retargeting'
DATA = PACKAGE / 'demo_data'
PLAN_ROOT = DATA / 'semantic_keyframes/g1_anatomy_v2_20260920'
B4 = 'uniform2_semantic_weight_full_event_transition_truncated_budget'
FORCE_AFFORDANCE = 'semantic_b4_force_affordance'
FORCE_AFFORDANCE_POINT_WEIGHT = 2000.0
FORCE_AFFORDANCE_NORMAL_WEIGHT = 2.0
COMPLETED = {'sub3_largebox_003', 'sub10_largebox_089'}
EXTRA_OMOMO_TASK_OBJECTS = {
    'sub16_largebox_007': 'largebox',
    'sub17_floorlamp_026': 'floorlamp',
}
METRICS = ['global_exact_mm', 'semantic_part_exact_mm', 'local_exact_mm', 'body_object_edge_exact_mm',
           'ordinary_mm', 'sqp_iterations', 'solver_calls', 'wall_time_s', 'optimization_time_s',
           'end_to_end_retarget_time_s',
           'max_body_box_penetration_mm', 'penetration_fraction', 'mean_positive_penetration_mm',
           'sliding_fraction', 'toe_sliding_velocity_m_s', 'contact_preservation']
SEMANTIC_METHODS = {'semantic_b4', FORCE_AFFORDANCE}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_force_affordance_artifact(
    path,
    *,
    official_source,
    semantic_plan,
    scene=None,
):
    path = Path(path).resolve()
    official_source = Path(official_source).resolve()
    semantic_plan = Path(semantic_plan).resolve()
    if official_source.parent.name != 'official_inputs':
        raise ValueError(
            'force-affordance retargeting must start from omomo/official_inputs'
        )
    with np.load(path, allow_pickle=False) as payload:
        forbidden = {'qpos', 'robot_qpos', 'retargeted_qpos'}.intersection(payload.files)
        if forbidden:
            raise ValueError(
                f'force-affordance artifact contains robot trajectory arrays: {sorted(forbidden)}'
            )
        if 'metadata_json' not in payload.files:
            raise ValueError('force-affordance artifact is missing metadata_json')
        metadata = json.loads(str(payload['metadata_json'].item()))
        active = np.asarray(payload['active'], dtype=bool)
    if metadata.get('no_retargeted_robot_trajectory_input') is not True:
        raise ValueError(
            'force-affordance artifact must prove no retargeted trajectory input'
        )
    if metadata.get('activation_mode') != 'force_affordance_interval':
        raise ValueError('force-affordance artifact has the wrong activation mode')
    expected_bodies = {
        'left_hand': 'left_rubber_hand_link',
        'right_hand': 'right_rubber_hand_link',
    }
    if metadata.get('force_affordance_robot_bodies') != expected_bodies:
        raise ValueError(
            'force-affordance artifact must use the original rubber-hand bodies'
        )
    interval = metadata.get('interval_derivation')
    if (
        not isinstance(interval, dict)
        or interval.get('schema') != 'holosoma.force_affordance_interval.v1'
        or interval.get('policy', {}).get('manual_frame_overrides') is not False
    ):
        raise ValueError(
            'force-affordance artifact requires a script-derived interval with '
            'manual overrides disabled'
        )
    start_frame = int(metadata.get('grip_start', -1))
    end_frame = int(metadata.get('end_frame', -1))
    active_frames = np.flatnonzero(np.any(active, axis=1))
    if (
        not len(active_frames)
        or int(active_frames[0]) != start_frame
        or int(active_frames[-1]) != end_frame
        or np.any(~np.all(active[start_frame : end_frame + 1], axis=1))
    ):
        raise ValueError('force-affordance active mask differs from its derived interval')
    inputs = metadata.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('force-affordance artifact is missing source inputs')
    expected = {
        'official_smpl_sha256': digest(official_source),
        'semantic_plan_sha256': digest(semantic_plan),
    }
    mismatches = {
        key: (inputs.get(key), value)
        for key, value in expected.items()
        if inputs.get(key) != value
    }
    interval_path = Path(str(inputs.get('interval_plan', ''))).resolve()
    if (
        not interval_path.is_file()
        or inputs.get('interval_plan_sha256') != digest(interval_path)
    ):
        mismatches['interval_plan_sha256'] = (
            inputs.get('interval_plan_sha256'),
            digest(interval_path) if interval_path.is_file() else None,
        )
    if mismatches:
        raise ValueError(f'force-affordance source mismatch: {mismatches}')
    if scene is not None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(Path(scene).resolve()))
        for body_name in expected_bodies.values():
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_name,
            )
            if body_id < 0:
                raise ValueError(f'force-affordance hand is absent: {body_name}')
            collision_geoms = [
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) == int(body_id)
                and (
                    int(model.geom_contype[geom_id]) != 0
                    or int(model.geom_conaffinity[geom_id]) != 0
                )
            ]
            if not collision_geoms:
                raise ValueError(
                    f'force-affordance hand has no collision geom: {body_name}'
                )
        for forbidden_body in ('left_sphere_hand_link', 'right_sphere_hand_link'):
            if (
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    forbidden_body,
                )
                >= 0
            ):
                raise ValueError(
                    'force-affordance scene replaces the original hand: '
                    f'{forbidden_body}'
                )
    return metadata


def weights():
    from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
    result = dict(SemanticRetargetingConfig().body_weight_multiplier)
    for side in ('left', 'right'):
        for part in ('hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist'):
            result[f'{side}_{part}'] = 2.0
    result['waist'] = 2.0
    return result


def omomo_object_name(task):
    if task in TASK_OBJECTS:
        return TASK_OBJECTS[task]
    return EXTRA_OMOMO_TASK_OBJECTS[task]


def make_job(
    dataset,
    task,
    method,
    output,
    plan_root=None,
    omomo_fallback_root=None,
    omomo_input_root=None,
    activate_obj_non_penetration=True,
    active_pair_nonpenetration_refinement=False,
    active_pair_acceptance_tolerance=0.01,
    contact_targets=None,
    contact_objective_weight=200.0,
    contact_normal_weight=0.5,
    semantic_plan=None,
    omomo_scene=None,
):
    plan_root = Path(plan_root) if plan_root is not None else PLAN_ROOT
    review_path = plan_root / 'PLAN_REVIEW_STATUS.json'
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review.get('block_new_runs'):
            raise ValueError(f'Semantic plan generation requires review: {review.get("reason")}; see {review_path}')
    plan_source = (
        Path(semantic_plan).resolve()
        if semantic_plan is not None
        else plan_root / dataset / task / 'semantic_plan.json'
    )
    plan_data = json.loads(plan_source.read_text())
    if plan_data.get('schema') == 'holosoma.trajectory_event_program.v1':
        approval = plan_data.get('generation_metadata', {})
        if not approval.get('approved_for_retargeting') or plan_data.get('unresolved_events') or not plan_data.get('events'):
            raise ValueError(f'{task}: fresh event plan has not passed the trajectory review')
    rules_path = plan_root / 'task_body_constraints.json'
    rules = json.loads(rules_path.read_text()) if rules_path.exists() else {}
    expected_constraint = rules.get(f'{dataset}/{task}')
    if expected_constraint:
        plan_data = json.loads(plan_source.read_text())
        if plan_data.get('generation_metadata', {}).get('body_part_constraints') != expected_constraint:
            raise ValueError(f'{task}: semantic plan does not match task body constraints')
        for event in plan_data['events']:
            parts = set(event['body_parts'])
            for region in ('hand', 'wrist', 'elbow', 'shoulder'):
                if (f'left_{region}' in parts) != (f'right_{region}' in parts):
                    raise ValueError(f'{task}/{event["event"]}: missing bilateral {region}')
    frozen = (
        output / 'inputs/plans' / dataset / task / 'semantic_plan.json'
        if semantic_plan is not None
        else DATA / 'semantic_keyframes/frozen' / output.name / dataset / task / 'semantic_plan.json'
    )
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != plan_source.read_bytes():
        raise ValueError(f'Frozen plan changed: {frozen}; use a new experiment name')
    if not frozen.exists():
        frozen.write_bytes(plan_source.read_bytes())
    if dataset == 'omomo':
        object_name = omomo_object_name(task)
        if omomo_input_root is not None:
            data_root = Path(omomo_input_root)
        else:
            data_root = DATA / 'omomo/official_inputs'
        if omomo_input_root is None and not (data_root / f'{task}.pt').is_file():
            data_root = (Path(omomo_fallback_root) if omomo_fallback_root is not None
                         else DATA / 'retarget_inputs/omomo_batch/input')
        data_root = data_root.resolve()
        if omomo_scene is not None:
            scene = Path(omomo_scene).resolve()
        elif object_name == 'largetable':
            # Validated shared compound geometry; same asset for both methods.
            scene = DATA / 'retarget_inputs/sub1_largetable_028_officialpt_compound_20260917/input/scenes/g1_29dof_w_largetable.xml'
        else:
            scene = data_root / 'scenes' / f'g1_29dof_w_{object_name}.xml'
        source = data_root / f'{task}.pt'
    else:
        data_root, scene = DATA / 'lafan', None
        source = data_root / f'{task}.npy'
    run = output / dataset / task / method
    command = [sys.executable, '-m', 'holosoma_retargeting.examples.robot_retarget',
               '--task-type', 'object_interaction' if dataset == 'omomo' else 'robot_only',
               '--task-name', task, '--data-format', 'smplh' if dataset == 'omomo' else 'lafan',
               '--data-path', str(data_root), '--save-dir', str(run),
               '--retargeter.foot-sticking-tolerance', '0.001',
               '--semantic.mode', 'original' if method == 'original' else B4,
               '--semantic.profile-dir', str(run / 'profile')]
    if dataset == 'omomo':
        command += ['--task-config.object-name', object_name, '--task-config.scene-xml-file', str(scene)]
    if method in SEMANTIC_METHODS:
        command = [sys.executable, str(Path(__file__).resolve()), '--execute-retarget', str(run / 'job.json')]
    sources = {str(p): digest(p) for p in (source, frozen, *([scene] if scene else []))}
    job = dict(dataset=dataset, task=task, method=method, run=str(run), input_root=str(data_root),
               scene=str(scene) if scene else None, plan=str(frozen), sources=sources, command=command)
    if method in SEMANTIC_METHODS:
        job['semantic_weights'] = weights()
        job['geometry_projection'] = False
        if not activate_obj_non_penetration:
            job['activate_obj_non_penetration'] = False
        if active_pair_nonpenetration_refinement:
            job['active_pair_nonpenetration_refinement'] = True
            job['active_pair_max_iterations'] = 10
            job['active_pair_acceptance_tolerance'] = float(active_pair_acceptance_tolerance)
            job['active_pair_prediction_margin'] = 0.012
        if method == FORCE_AFFORDANCE:
            if contact_targets is None:
                raise ValueError('anchor-force retargeting requires contact_targets')
            contact_targets = Path(contact_targets).resolve()
            validate_force_affordance_artifact(
                contact_targets,
                official_source=source,
                semantic_plan=plan_source,
                scene=scene,
            )
            job['contact_targets'] = str(contact_targets)
            job['contact_objective_weight'] = float(contact_objective_weight)
            job['contact_normal_weight'] = float(contact_normal_weight)
            job['sources'][str(contact_targets)] = digest(contact_targets)
    job['signature'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    return job


def evaluate(job, result):
    from holosoma_retargeting.semantic_keyframes.precision import evaluate_precision, load_precision_payload
    from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection
    from holosoma_retargeting.evaluation.eval_retargeting import RetargetingEvaluator, create_task_constants
    from holosoma_retargeting.config_types.robot import RobotConfig
    from holosoma_retargeting.config_types.data_type import MotionDataConfig
    os.chdir(PACKAGE)
    run = Path(job['run'])
    plan = load_semantic_plan_projection(job['plan'])
    precision_payload = load_precision_payload(result)
    ev = evaluate_precision(precision_payload, plan, critical_event_names=None)
    profile = json.loads((run / 'profile/profile.json').read_text())
    profile_payload = profile
    profile = profile_payload.get('summary', profile_payload)
    row = {k: job[k] for k in ('dataset', 'task', 'method', 'signature')}
    row['source_vertices_sha256'] = hashlib.sha256(precision_payload.source_vertices.tobytes()).hexdigest()
    row['source_adjacency_sha256'] = hashlib.sha256(precision_payload.adjacency.tobytes()).hexdigest()
    row.update(status='ok', result=str(result), global_exact_mm=1000*ev.keyframe_global['exact'],
               semantic_part_exact_mm=1000*ev.semantic_part['exact'], local_exact_mm=1000*ev.semantic_local['exact'],
               body_object_edge_exact_mm=1000*ev.semantic_edge['exact'] if ev.semantic_edge and job['dataset']=='omomo' else None,
               ordinary_mm=1000*ev.ordinary['mean'], sqp_iterations=profile['total_actual_iterations'],
               solver_calls=profile.get('total_convex_solver_calls'), wall_time_s=profile['total_wall_time'],
               optimization_time_s=profile['optimization_wall_time'])
    contact_frames = [
        frame for frame in profile_payload.get('frames', [])
        if int(frame.get('contact_active_count', 0) or 0) > 0
    ]
    row['contact_keyframes'] = [int(frame['frame_idx']) for frame in contact_frames]
    row['contact_active_frames'] = row['contact_keyframes']
    row['palm_anchor_error_at_tc_mm'] = (
        1000 * float(np.mean([frame['contact_error_mean_m'] for frame in contact_frames]))
        if contact_frames else None
    )
    normal_errors = [
        float(value)
        for frame in contact_frames
        for value in (
            frame.get('contact_normal_error_deg')
            or frame.get('palm_normal_error_deg')
            or {}
        ).values()
    ]
    row['palm_normal_error_at_tc_deg'] = (
        float(np.mean(normal_errors)) if normal_errors else None
    )
    surface_gaps = []
    for frame in contact_frames:
        for part in frame.get('contact_active_parts', []):
            value = frame.get(f"{part.removesuffix('_hand')}_hand_box_signed_distance")
            if value is not None:
                surface_gaps.append(float(value))
    row['hand_object_surface_gap_at_tc_mm'] = (
        1000 * float(np.mean(surface_gaps)) if surface_gaps else None
    )
    point_errors_by_part = {}
    gamma_values_by_part = {}
    for frame in profile_payload.get('frames', []):
        for part, value in (
            frame.get('force_affordance_point_error_m') or {}
        ).items():
            point_errors_by_part.setdefault(part, []).append(float(value))
        for part, value in (frame.get('contact_gamma_by_part') or {}).items():
            gamma_values_by_part.setdefault(part, []).append(float(value))
    row['anchor_force_point_error_stats_mm'] = {
        part: {
            'frames': len(values),
            'mean': 1000 * float(np.mean(values)),
            'median': 1000 * float(np.median(values)),
            'p95': 1000 * float(np.quantile(values, .95)),
            'max': 1000 * float(np.max(values)),
        }
        for part, values in point_errors_by_part.items()
    }
    row['contact_gamma_stats'] = {
        part: {'frames': len(values), 'mean': float(np.mean(values)), 'max': float(np.max(values))}
        for part, values in gamma_values_by_part.items()
    }
    row['contact_normal_error_mean_deg'] = (
        float(np.mean(normal_errors)) if normal_errors else None
    )
    audit = run / 'profile/penetration_pairs.csv'
    if audit.exists() and job['dataset']=='omomo':
        with audit.open() as stream:
            audit_rows = list(csv.DictReader(stream))
        other_depths = [float(r['penetration_depth'])*1000 for r in audit_rows if r.get('pair_type')=='other-body-box']
        contact_depths = [float(r['penetration_depth'])*1000 for r in audit_rows if r.get('pair_type')=='hand-box']
        illegal_depths = [float(r['penetration_depth'])*1000 for r in audit_rows if r.get('illegal_penetration')=='True']
        row['max_body_box_penetration_mm'] = max(other_depths, default=0.0)
        row['max_expected_contact_penetration_mm'] = max(contact_depths, default=0.0)
        row['max_illegal_penetration_mm'] = max(illegal_depths, default=0.0)
    else:
        row['max_body_box_penetration_mm'] = None
        row['max_expected_contact_penetration_mm'] = None
        row['max_illegal_penetration_mm'] = None
    with np.load(result, allow_pickle=False) as data:
        row['frames'] = len(data['qpos'])
        if job['dataset']=='lafan':
            expected = len(np.load(Path(job['input_root']) / f"{job['task']}.npy", mmap_mode='r'))
        else:
            import torch
            expected = len(torch.load(Path(job['input_root']) / f"{job['task']}.pt", weights_only=False, map_location='cpu'))
        if row['frames'] != expected:
            raise ValueError(f"Partial trajectory: {row['frames']}/{expected}")
        row['unweighted_frame_mean_mm'] = float(np.mean(data['unweighted_vertex_residuals']))*1000
    precision = {k:getattr(ev,k) for k in ('ordinary','keyframe_global','semantic_part','semantic_local','semantic_edge','event_rows','body_part_rows')}
    write(run/'precision.json', precision)
    obj = omomo_object_name(job['task']) if job['dataset']=='omomo' else 'ground'
    constants = create_task_constants(RobotConfig(robot_type='g1'), MotionDataConfig(data_format='smplh' if job['dataset']=='omomo' else 'lafan',robot_type='g1'), object_name=obj)
    if job['scene']:
        constants.SCENE_XML_FILE = job['scene']
    evaluator = RetargetingEvaluator(constants.ROBOT_URDF_FILE, getattr(constants,'OBJECT_URDF_FILE',None), obj,
                                    constants.DEMO_JOINTS, constants.JOINTS_MAPPING, visualize=False, constants=constants)
    fn = evaluator.evaluate_trajectory if job['dataset']=='omomo' else evaluator.evaluate_robot_only_trajectory
    m = fn(job['task'], str(result), job['input_root'])
    if m is None:
        raise ValueError('Official evaluator returned None')
    mean = lambda values: float(np.mean(values)) if np.asarray(values).size else 0.0
    row.update(penetration_fraction=float(m['penetration_duration']),
               mean_positive_penetration_mm=1000*mean(m['penetration_max_depths']),
               sliding_fraction=float(m['sliding_duration']), toe_sliding_velocity_m_s=mean(m['max_toe_sliding_velocities']),
               contact_preservation=float(m['contact_preservation']) if 'contact_preservation' in m else None)
    write(run / 'native_penetration_raw.json', dict(
        threshold_m=0.01, duration=float(m['penetration_duration']),
        penetration_max_depths_m=np.asarray(m['penetration_max_depths']).tolist(),
        foot_sliding_threshold_native=0.01,
        sliding_duration=float(m['sliding_duration']),
        max_toe_sliding_velocities_native=np.asarray(m['max_toe_sliding_velocities']).tolist()))
    return row


def worker(job):
    run = Path(job['run']); run.mkdir(parents=True,exist_ok=True)
    for p, sha in job['sources'].items():
        if digest(Path(p)) != sha:
            raise ValueError(f'Input changed during experiment: {p}')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'src/holosoma_retargeting'), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with (run/'retarget.log').open('w') as log:
        retarget_started = time.perf_counter()
        p = subprocess.run(job['command'], cwd=PACKAGE, env=env, stdout=log, stderr=subprocess.STDOUT)
        retarget_elapsed = time.perf_counter() - retarget_started
    result = run / (job['task']+'_original.npz' if job['dataset']=='omomo' else job['task']+'.npz')
    if p.returncode or not result.exists():
        raise RuntimeError(f'Retarget failed (exit {p.returncode}); '+ '\n'.join((run/'retarget.log').read_text(errors='replace').splitlines()[-12:]))
    row = evaluate(job,result)
    row['end_to_end_retarget_time_s'] = retarget_elapsed
    return row


def publish(output, jobs, rows, active):
    by={(r['dataset'],r['task'],r['method']):r for r in rows}
    pairs=[]
    for ds, task in dict.fromkeys((j['dataset'],j['task']) for j in jobs):
        a=by.get((ds,task,'original'))
        if not a or a['status']!='ok':continue
        for method in sorted(SEMANTIC_METHODS):
            b=by.get((ds,task,method))
            if not b or b['status']!='ok':continue
            if a.get('source_vertices_sha256') != b.get('source_vertices_sha256') or a.get('source_adjacency_sha256') != b.get('source_adjacency_sha256'):
                continue  # Incomparable source geometry must never enter paired aggregates.
            delta={k: 100*(b[k]/a[k]-1) if a.get(k) is not None and b.get(k) is not None and a[k]!=0 else None for k in METRICS}
            pairs.append(dict(dataset=ds,task=task,comparison_method=method,original=a,semantic=b,relative_delta_percent=delta))
    payload=dict(updated_at=time.time(), requested_runs=len(jobs), finished_runs=len(rows), active=active, rows=rows, paired=pairs)
    write(output/'comparison.json',payload)
    lines=['# Current semantic plan: Original vs semantic methods','',f'Finished runs: {len(rows)}/{len(jobs)}; complete paired comparisons: {len(pairs)}.','',
           'Same native input frames, collision scene and physical constraints for all methods. No stride-20 LAFAN fallback; no disabled foot/object constraints. Original is rerun, not an older score. B4 uses Uniform-2, transition-truncated full-event weighting and exact-trigger budget 4. Anchor-force adds precomputed source-derived palm anchors and normals while preserving the same B4 SQP budget. Metrics use the same frozen plan and unweighted residuals. LAFAN has no body-object edge metric; ground edges are not mislabeled as object edges.', '',
           'Runtime includes concurrent CPU load; SQP/solver counts are the more stable compute comparison. Precision is mm; penetration/sliding duration metrics are frame fractions. Cost under differently weighted objectives is not used as a precision metric.', '',
           '| Dataset/task + method | Original Part Exact (mm) | Semantic Part Exact (mm) | Change | Original SQP | Semantic SQP |',
           '|---|---:|---:|---:|---:|---:|']
    for p in pairs:
        a,b=p['original'],p['semantic'];d=p['relative_delta_percent']['semantic_part_exact_mm']
        lines.append(f"| {p['dataset']}/{p['task']} ({p['comparison_method']}) | {a['semantic_part_exact_mm']:.3f} | {b['semantic_part_exact_mm']:.3f} | {d:+.2f}% | {a['sqp_iterations']} | {b['sqp_iterations']} |")
    lines += ['', '## Failed runs (constraints are not relaxed)', '']
    for r in rows:
        if r['status']!='ok':lines.append(f"- {r['dataset']}/{r['task']}/{r['method']}: see `{r['run']}/worker.log` and `retarget.log`.")
    lines += ['', '## Running', '', *[f"- {x}" for x in active], '']
    (output/'COMPARISON.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison')
    ap.add_argument('--tasks',nargs='+')
    ap.add_argument('--plan-root',type=Path,help='Explicit reviewed plan set; legacy default remains blocked')
    ap.add_argument('--semantic-plan', type=Path,
                    help='Explicit semantic plan for the requested single task.')
    ap.add_argument('--omomo-scene', type=Path,
                    help='Explicit local MuJoCo scene; trajectory still comes from official_inputs.')
    ap.add_argument('--omomo-fallback-input-root', type=Path,
                    help='Input root for registered OMOMO tasks absent from official_inputs')
    ap.add_argument('--omomo-input-root', type=Path,
                    help='Explicit OMOMO input root, overriding official_inputs selection.')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument(
        '--methods', nargs='+',
        choices=('original', 'semantic_b4', FORCE_AFFORDANCE),
        default=['original', 'semantic_b4'],
    )
    ap.add_argument(
        '--disable-object-nonpenetration',
        action='store_true',
        help='Run Semantic B4 with activate_obj_non_penetration=False.',
    )
    ap.add_argument(
        '--active-pair-nonpenetration-refinement',
        action='store_true',
        help='Run at most 10 extra SQP steps when active-pair penetration exceeds 10 mm.',
    )
    ap.add_argument('--active-pair-acceptance-tolerance', type=float, default=0.01)
    ap.add_argument('--worker',type=Path)
    ap.add_argument('--execute-retarget',type=Path)
    ap.add_argument('--contact-targets', type=Path,
                    help='Opt-in semantic surface-contact artifact (single OMOMO task only).')
    ap.add_argument('--contact-objective-weight', type=float)
    ap.add_argument('--contact-normal-weight', type=float)
    args=ap.parse_args()
    if args.execute_retarget:
        from holosoma_retargeting.config_types.retargeting import RetargetingConfig
        from holosoma_retargeting.config_types.retargeter import RetargeterConfig
        from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
        from holosoma_retargeting.config_types.task import TaskConfig
        from holosoma_retargeting.examples.robot_retarget import main as retarget
        job = json.loads(args.execute_retarget.read_text())
        task_cfg = TaskConfig(object_name=omomo_object_name(job['task']), scene_xml_file=job['scene']) if job['dataset']=='omomo' else TaskConfig()
        cfg = RetargetingConfig(task_type='object_interaction' if job['dataset']=='omomo' else 'robot_only',
            task_name=job['task'], data_format='smplh' if job['dataset']=='omomo' else 'lafan',
            data_path=Path(job['input_root']), save_dir=Path(job['run']), task_config=task_cfg,
            retargeter=RetargeterConfig(
                foot_sticking_tolerance=0.001,
                activate_obj_non_penetration=job.get('activate_obj_non_penetration', True),
            ),
            semantic=SemanticRetargetingConfig(
                mode=B4,
                exact_trigger_budget=4,
                contact_targets_path=(
                    Path(job['contact_targets']) if job.get('contact_targets') else None
                ),
                contact_objective_weight=job.get('contact_objective_weight', 200.0),
                contact_normal_weight=job.get('contact_normal_weight', 0.5),
                geometry_projection=job.get('geometry_projection', False),
                geometry_projection_max_iterations=job.get('geometry_projection_max_iterations', 10),
                active_pair_nonpenetration_refinement=job.get(
                    'active_pair_nonpenetration_refinement', False
                ),
                active_pair_max_iterations=job.get('active_pair_max_iterations', 10),
                active_pair_acceptance_tolerance=job.get(
                    'active_pair_acceptance_tolerance', 0.01
                ),
                active_pair_prediction_margin=job.get('active_pair_prediction_margin', 0.012),
                semantic_keyframe_path=Path(job['plan']), profile_dir=Path(job['run'])/'profile',
                body_weight_multiplier=job['semantic_weights']))
        retarget(cfg)
        return
    if args.worker:
        job=json.loads(args.worker.read_text())
        try: row=worker(job)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            row={k:job[k] for k in ('dataset','task','method','signature','run')}
            row.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write(Path(job['run'])/'metrics.json',row)
        return
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    tasks=[('omomo',t) for t in TASK_OBJECTS if t not in COMPLETED]+[('lafan',t) for t in LAFAN_TASKS]
    if args.tasks:
        omomo_tasks = set(TASK_OBJECTS) | set(EXTRA_OMOMO_TASK_OBJECTS)
        known = omomo_tasks | set(LAFAN_TASKS)
        unknown=set(args.tasks)-known
        if unknown:ap.error(f'Unknown tasks: {unknown}')
        tasks=[('omomo' if task in omomo_tasks else 'lafan', task) for task in args.tasks]
    if args.contact_targets is not None:
        if len(tasks) != 1 or tasks[0][0] != 'omomo':
            ap.error('--contact-targets requires exactly one OMOMO task')
        if FORCE_AFFORDANCE not in args.methods:
            ap.error('--contact-targets requires semantic_b4_force_affordance')
    if FORCE_AFFORDANCE in args.methods and args.contact_targets is None:
        ap.error('semantic_b4_force_affordance requires --contact-targets')
    if args.semantic_plan is not None and len(tasks) != 1:
        ap.error('--semantic-plan requires exactly one task')
    jobs=[make_job(
              d,t,m,output,args.plan_root,args.omomo_fallback_input_root,args.omomo_input_root,
              activate_obj_non_penetration=not args.disable_object_nonpenetration,
              active_pair_nonpenetration_refinement=args.active_pair_nonpenetration_refinement,
              active_pair_acceptance_tolerance=args.active_pair_acceptance_tolerance,
              contact_targets=args.contact_targets if m == FORCE_AFFORDANCE else None,
              contact_objective_weight=(
                  FORCE_AFFORDANCE_POINT_WEIGHT
                  if m == FORCE_AFFORDANCE and args.contact_objective_weight is None
                  else 200.0
                  if args.contact_objective_weight is None
                  else args.contact_objective_weight
              ),
              contact_normal_weight=(
                  FORCE_AFFORDANCE_NORMAL_WEIGHT
                  if m == FORCE_AFFORDANCE and args.contact_normal_weight is None
                  else 0.5
                  if args.contact_normal_weight is None
                  else args.contact_normal_weight
              ),
              semantic_plan=args.semantic_plan,
              omomo_scene=args.omomo_scene,
          )
          for d,t in tasks for m in args.methods]
    write(output/'manifest.json',dict(
        jobs=jobs,
        weights=weights(),
        workers=args.workers,
        physical_constraints=(
            'object nonpenetration disabled by explicit ablation; 1mm foot sticking; step size 0.2'
            if args.disable_object_nonpenetration else
            'original defaults; 1mm foot sticking and penetration tolerance; step size 0.2'
        ),
        activate_obj_non_penetration=not args.disable_object_nonpenetration,
        active_pair_nonpenetration_refinement=args.active_pair_nonpenetration_refinement,
        lafan_stride=1,
    ))
    rows=[]
    def launch(job):
        run=Path(job['run']);run.mkdir(parents=True,exist_ok=True)
        path=run/'job.json';write(path,job)
        cached=run/'metrics.json'
        if cached.exists():
            row=json.loads(cached.read_text())
            if row.get('signature')==job['signature'] and row['status']=='ok':return row
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src/holosoma_retargeting')+':'+str(ROOT/'tools'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (run/'worker.log').open('w') as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads(cached.read_text())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(launch,j):j for j in jobs}
        while pending:
            done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
            for f in done:
                j=pending.pop(f)
                try:row=f.result()
                except Exception as exc:row={k:j[k] for k in ('dataset','task','method','run')};row.update(status='failed',error=str(exc))
                rows.append(row)
                print(json.dumps({k:row[k] for k in ('dataset','task','method','status')}),flush=True)
            active=['/'.join((j['dataset'],j['task'],j['method'])) for f,j in pending.items() if f.running()]
            publish(output,jobs,rows,active)
    print(json.dumps({'finished':len(rows),'ok':sum(r['status']=='ok' for r in rows),'output':str(output)}),flush=True)


if __name__=='__main__':main()
