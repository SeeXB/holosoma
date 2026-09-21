#!/usr/bin/env python3
"""Fresh whole-clip generation and current-run self-reflection with live status."""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from pilot_trajectory_event_plans import DATA, ROOT, load_env_file
from pilot_trajectory_event_plans import VERSION as SINGLE_VERSION
from pilot_trajectory_event_plans import generate_once as generate_single
from prepare_batch_retarget_inputs import TASK_OBJECTS
from prepare_lafan_batch_inputs import TASKS as LAFAN_TASKS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--dataset', choices=('omomo', 'lafan', 'all'), default='omomo')
    parser.add_argument('--sr-rounds', type=int, choices=range(7), default=6)
    parser.add_argument('--syntax-repair-rounds', type=int, choices=range(4), default=1)
    parser.add_argument('--workflow', choices=('agents', 'single'), default='agents')
    args = parser.parse_args()
    if args.workflow == 'agents':
        from generate_multi_agent_event_plans import VERSION, generate_once
    else:
        VERSION, generate_once = SINGLE_VERSION, generate_single
    if not args.prepare_only:
        load_env_file(ROOT / 'src/holosoma_retargeting/.env')
    tasks = [('omomo', task) for task in TASK_OBJECTS if task not in {'sub3_largebox_003','sub10_largebox_089'}]
    tasks += [('lafan', task) for task in LAFAN_TASKS]
    if args.dataset != 'all':
        tasks = [(ds,task) for ds,task in tasks if ds == args.dataset]
    if args.tasks:
        if set(args.tasks) - {task for _,task in tasks}:
            parser.error('Unknown task')
        tasks = [(ds,task) for ds,task in tasks if task in args.tasks]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    audit = ROOT / 'exp/semantic_plans' / VERSION / 'batches' / stamp
    audit.mkdir(parents=True, exist_ok=False)
    rules = json.loads((DATA / 'semantic_keyframes/g1_anatomy_v2_20260920/task_body_constraints.json').read_text())
    rows = []
    quota_exhausted = threading.Event()

    def publish():
        payload = dict(total=len(tasks), finished=len(rows), done=len(rows)==len(tasks),
                       workflow=args.workflow,
                       fresh_initial_request=True, max_self_reflection_rounds=args.sr_rounds,
                       max_syntax_repair_calls=(args.syntax_repair_rounds if args.workflow == 'agents' else 0),
                       results=sorted(rows,key=lambda r:(r['dataset'],r['task'])))
        path=audit/'summary.json'
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(payload,indent=2)+'\n');temp.replace(path)
        lines=['# Fresh event generation', '',f'Finished {len(rows)}/{len(tasks)} tasks.', '',
               '|Dataset/task|Status|Key-frame fraction|', '|---|---|---:|']
        for row in payload['results']:
            coverage=row.get('coverage',{}).get('fraction')
            lines.append(f"|{row['dataset']}/{row['task']}|{row['status']}|{coverage if coverage is not None else '—'}|")
        (audit/'MONITOR.md').write_text('\n'.join(lines)+'\n')

    def run(ds,task):
        if quota_exhausted.is_set():
            return dict(dataset=ds, task=task, status='blocked_quota', vlm_calls=0)
        base=DATA/'omomo/bundles'/task
        kwargs = dict(bundle=base/'input/omomo_gt_sequence.npz',
                      video=base/'videos'/f'{task}_rerender.mp4', constraints=rules.get(ds+'/'+task,{}),
                      output_root=DATA/'semantic_keyframes'/VERSION/ds/task,
                      audit_root=ROOT/'exp/semantic_plans'/VERSION/ds/task,
                      prepare_only=args.prepare_only,
                      sr_rounds=args.sr_rounds,
                      lafan_source=DATA/'lafan'/f'{task}.npy' if ds=='lafan' else None)
        if args.workflow == 'agents':
            kwargs['syntax_repair_rounds'] = args.syntax_repair_rounds
        row=generate_once(**kwargs)
        if row.get('error_type') == 'VLMQuotaError':
            quota_exhausted.set()
        return dict(dataset=ds,task=task,**row)
    publish()
    print(str(audit),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run,ds,task):(ds,task) for ds,task in tasks}
        for future in as_completed(futures):
            ds,task=futures[future]
            try: row=future.result()
            except Exception as exc: row=dict(dataset=ds,task=task,status='preparation_failed',error_type=type(exc).__name__)
            rows.append(row);publish()
            print(json.dumps({k:row[k] for k in ('dataset','task','status')}),flush=True)
    return int(any(row['status'] not in ('prepared_only', 'requires_visual_and_signal_review') for row in rows))


if __name__=='__main__':
    raise SystemExit(main())
