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


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    activate_obj_non_penetration=True,
    active_pair_nonpenetration_refinement=False,
):
    plan_root = Path(plan_root) if plan_root is not None else PLAN_ROOT
    review_path = plan_root / 'PLAN_REVIEW_STATUS.json'
    if review_path.exists():
        review = json.loads(review_path.read_text())
        if review.get('block_new_runs'):
            raise ValueError(f'Semantic plan generation requires review: {review.get("reason")}; see {review_path}')
    plan_source = plan_root / dataset / task / 'semantic_plan.json'
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
    frozen = DATA / 'semantic_keyframes/frozen' / output.name / dataset / task / 'semantic_plan.json'
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists() and frozen.read_bytes() != plan_source.read_bytes():
        raise ValueError(f'Frozen plan changed: {frozen}; use a new experiment name')
    if not frozen.exists():
        frozen.write_bytes(plan_source.read_bytes())
    if dataset == 'omomo':
        object_name = omomo_object_name(task)
        data_root = DATA / 'omomo/official_inputs'
        if not (data_root / f'{task}.pt').is_file():
            data_root = (Path(omomo_fallback_root) if omomo_fallback_root is not None
                         else DATA / 'retarget_inputs/omomo_batch/input')
        data_root = data_root.resolve()
        if object_name == 'largetable':
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
    if method == 'semantic_b4':
        command = [sys.executable, str(Path(__file__).resolve()), '--execute-retarget', str(run / 'job.json')]
    sources = {str(p): digest(p) for p in (source, frozen, *([scene] if scene else []))}
    job = dict(dataset=dataset, task=task, method=method, run=str(run), input_root=str(data_root),
               scene=str(scene) if scene else None, plan=str(frozen), sources=sources, command=command)
    if method == 'semantic_b4':
        job['semantic_weights'] = weights()
        job['geometry_projection'] = False
        if not activate_obj_non_penetration:
            job['activate_obj_non_penetration'] = False
        if active_pair_nonpenetration_refinement:
            job['active_pair_nonpenetration_refinement'] = True
            job['active_pair_max_iterations'] = 10
            job['active_pair_acceptance_tolerance'] = 0.01
            job['active_pair_prediction_margin'] = 0.012
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
    profile = profile.get('summary', profile)
    row = {k: job[k] for k in ('dataset', 'task', 'method', 'signature')}
    row['source_vertices_sha256'] = hashlib.sha256(precision_payload.source_vertices.tobytes()).hexdigest()
    row['source_adjacency_sha256'] = hashlib.sha256(precision_payload.adjacency.tobytes()).hexdigest()
    row.update(status='ok', result=str(result), global_exact_mm=1000*ev.keyframe_global['exact'],
               semantic_part_exact_mm=1000*ev.semantic_part['exact'], local_exact_mm=1000*ev.semantic_local['exact'],
               body_object_edge_exact_mm=1000*ev.semantic_edge['exact'] if ev.semantic_edge and job['dataset']=='omomo' else None,
               ordinary_mm=1000*ev.ordinary['mean'], sqp_iterations=profile['total_actual_iterations'],
               solver_calls=profile.get('total_convex_solver_calls'), wall_time_s=profile['total_wall_time'],
               optimization_time_s=profile['optimization_wall_time'])
    audit = run / 'profile/penetration_pairs.csv'
    if audit.exists() and job['dataset']=='omomo':
        with audit.open() as stream:
            depths = [float(r['penetration_depth'])*1000 for r in csv.DictReader(stream) if r.get('pair_type')=='other-body-box']
        row['max_body_box_penetration_mm'] = max(depths, default=0.0)
    else:
        row['max_body_box_penetration_mm'] = None
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
        a,b=(by.get((ds,task,m)) for m in ('original','semantic_b4'))
        if not a or not b or a['status']!='ok' or b['status']!='ok':continue
        if a.get('source_vertices_sha256') != b.get('source_vertices_sha256') or a.get('source_adjacency_sha256') != b.get('source_adjacency_sha256'):
            continue  # Incomparable source geometry must never enter paired aggregates.
        delta={k: 100*(b[k]/a[k]-1) if a.get(k) is not None and b.get(k) is not None and a[k]!=0 else None for k in METRICS}
        pairs.append(dict(dataset=ds,task=task,original=a,semantic_b4=b,relative_delta_percent=delta))
    payload=dict(updated_at=time.time(), requested_runs=len(jobs), finished_runs=len(rows), active=active, rows=rows, paired=pairs)
    write(output/'comparison.json',payload)
    lines=['# Current semantic plan: Original vs Semantic B4','',f'Finished runs: {len(rows)}/{len(jobs)}; complete paired comparisons: {len(pairs)}.','',
           'Same native input frames, collision scene and physical constraints for both methods. No stride-20 LAFAN fallback; no disabled foot/object constraints. Original is rerun, not an older score. B4: Uniform-2, transition-truncated full-event weighting, exact-trigger budget 4; existing hand multiplier 4, added anatomy multipliers 2. Metrics use the same frozen new plan and unweighted residuals. LAFAN has no body-object edge metric; ground edges are not mislabeled as object edges.', '',
           'Runtime includes concurrent CPU load; SQP/solver counts are the more stable compute comparison. Precision is mm; penetration/sliding duration metrics are frame fractions. Cost under differently weighted objectives is not used as a precision metric.', '',
           '| Dataset/task | Original Part Exact (mm) | B4 Part Exact (mm) | Change | Original SQP | B4 SQP |',
           '|---|---:|---:|---:|---:|---:|']
    for p in pairs:
        a,b=p['original'],p['semantic_b4'];d=p['relative_delta_percent']['semantic_part_exact_mm']
        lines.append(f"| {p['dataset']}/{p['task']} | {a['semantic_part_exact_mm']:.3f} | {b['semantic_part_exact_mm']:.3f} | {d:+.2f}% | {a['sqp_iterations']} | {b['sqp_iterations']} |")
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
    ap.add_argument('--omomo-fallback-input-root', type=Path,
                    help='Input root for registered OMOMO tasks absent from official_inputs')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--methods', nargs='+', choices=('original', 'semantic_b4'), default=['original', 'semantic_b4'])
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
    ap.add_argument('--worker',type=Path)
    ap.add_argument('--execute-retarget',type=Path)
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
            semantic=SemanticRetargetingConfig(mode=B4, exact_trigger_budget=4,
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
    jobs=[make_job(
              d,t,m,output,args.plan_root,args.omomo_fallback_input_root,
              activate_obj_non_penetration=not args.disable_object_nonpenetration,
              active_pair_nonpenetration_refinement=args.active_pair_nonpenetration_refinement,
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
