"""Serial intermediate-checkpoint evaluation using the production Paper-DR protocol.

Never stops or edits training. Missing checkpoints and insufficient GPU memory
are waited for. Run inside a persistent session with the hssim environment.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_three_group_experiment import eval_command, now, preflight, run_process, save_json
from scripts.analyze_sub10_largebox_089_final import summarize_run


def checkpoint_for(cfg, group, step):
    paths = list((Path(cfg['work_root']) / f'group{group}' / 'train').rglob(f'model_{step:05d}.pt'))
    if len(paths) > 1:
        raise ValueError(f'Ambiguous checkpoint: {paths}')
    # Avoid reading a checkpoint while the training process is still writing it.
    return paths[0] if paths and time.time() - paths[0].stat().st_mtime > 30 else None


def free_memory(gpu):
    value = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free', '--format=csv,noheader,nounits'
    ], text=True)
    return int(value.strip())


def canonical_hashes(hashes):
    # Data reorganization left compatibility symlinks. Compare the same assets
    # by resolved location while still requiring byte-for-byte equality.
    return {str(Path(k).resolve()) if Path(k).is_absolute() else k: v for k, v in hashes.items()}


def report(cfg, root, jobs):
    results = []
    for group, step in jobs:
        path = root / f'group{group}_{step:05d}' / 'result.json'
        if path.is_file():
            results.append(json.loads(path.read_text()))
    results.sort(key=lambda r: (r['group'], r['checkpoint_iteration']))
    payload = {
        'task': cfg['task'], 'updated': now(), 'completed': len(results), 'expected': len(jobs),
        'note': 'No exact 10k checkpoint exists; 8k and 12k bracket 10k.',
        'protocol': 'Paper-DR; seed 42; 32 envs, first 10 completed episodes/env including failures; '
                    'own training reference, frame 0, full (N-1)/fps horizon, no initial pose noise; '
                    'push interval 1-3s, linear 0.3m/s, angular 0.78rad/s; object thresholds 1m/pi/4; '
                    'other termination conditions unchanged; shape scaling unsupported.',
        'results': results,
    }
    save_json(root / 'checkpoint_results.json', payload)
    lines = [f"# {cfg['task']} intermediate checkpoint evaluation", '', payload['note'], '',
             payload['protocol'], '', f'Completed: {len(results)}/{len(jobs)}', '',
             'RMSE covers observed frames, including early failures; it does not imply full-clip success.', '',
             '| Group | Iteration | Success | Rate | 95% CI | Duration (s) | Object RMSE (m) | Orientation RMSE (deg) | Body RMSE (m) |',
             '|---|---:|---:|---:|---|---:|---:|---:|---:|']
    for r in results:
        e, ci = r['trajectory_errors'], r['clustered_95ci']
        lines.append(f"| {r['group']} | {r['checkpoint_iteration']} | {r['successes']}/{r['episodes_scored']} | "
                     f"{r['success_rate']:.2%} | {ci[0]:.2%}–{ci[1]:.2%} | {r['mean_episode_seconds']:.3f} | "
                     f"{e['object_pos_rmse_m']:.4f} | {e['object_ori_rmse_deg']:.2f} | {e['tracked_body_pos_rmse_m']:.4f} |")
    for r in results:
        lines += ['', f"Group {r['group']}, iteration {r['checkpoint_iteration']}: "
                  f"failures={r['exclusive_failure_reasons']}; pushes={r['push_events_in_scored_episodes']}."]
    (root / 'CHECKPOINT_EVAL_REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--eval-root', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, default=[8000, 12000, 20000])
    parser.add_argument('--min-free-mib', type=int, default=5000)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    hashes = preflight(cfg)
    frozen = json.loads((Path(cfg['work_root']) / 'input_sha256.json').read_text())
    if canonical_hashes(hashes) != canonical_hashes(frozen):
        raise ValueError('Production inputs changed since training launch')
    jobs = [(g, s) for s in args.steps for g in (1, 2, 3)]
    inventory = [{ 'group': g, 'iteration': s, 'checkpoint': str(checkpoint_for(cfg, g, s) or '')}
                 for g, s in jobs]
    if args.check_only:
        print(json.dumps({'input_hashes_unchanged': True, 'jobs': inventory}, indent=2))
        return
    root = args.eval_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'eval.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = {'config': cfg, 'steps': args.steps, 'input_sha256': hashes}
    manifest_path = root / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing evaluation manifest differs')
    save_json(manifest_path, manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.get('gpu', 0)), WANDB_MODE='disabled')
    save_json(root / 'upload_policy.json', {'evaluation_upload': False, 'reason': 'User requested local-only evaluation'})
    remaining = list(jobs)
    while remaining:
        progressed = False
        for group, step in list(remaining):
            checkpoint = checkpoint_for(cfg, group, step)
            if checkpoint is None:
                continue
            out = root / f'group{group}_{step:05d}'
            out.mkdir(exist_ok=True)
            state_path = out / 'state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            motion = cfg['original_motion'] if group == 1 else cfg['b4_motion']
            if not state.get('evaluation_complete'):
                if free_memory(cfg.get('gpu', 0)) < args.min_free_mib:
                    continue
                state.update(group=group, checkpoint_iteration=step, checkpoint=str(checkpoint),
                             checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                # Completed simulation data can be rescored after a postprocessing interruption.
                if not (out / 'rollout_all_envs.npz').exists() or state.get('evaluation_exit_code') != 0:
                    command = eval_command(cfg, checkpoint, out, motion)
                    # get_eval_config already selects logger:disabled. Appending
                    # that subcommand after its options causes a tyro conflict.
                    command += ['--logger.base-dir', out / 'logs']
                    run_process(command, out / 'eval.log', state, state_path, 'evaluation', env, timeout=7200)
                result = summarize_run(out, 10, step, {group: checkpoint.parent}, reference_file=Path(motion))
                if result['episodes_scored'] != 320 or result['num_envs'] != 32:
                    raise ValueError('Incomplete evaluation quota')
                save_json(out / 'result.json', result)
                state.update(evaluation_complete=True)
                save_json(state_path, state)
                report(cfg, root, jobs)
                print(f"{now()} group={group} iteration={step} SR={result['successes']}/320", flush=True)
            if not state.get('video_complete'):
                run_process([cfg['render_python'], 'scripts/render_clothesstand_checkpoint_eval.py',
                             '--actual', out / 'rollout.npz', '--reference', motion, '--scene', cfg['scene_xml'],
                             '--label', f'Group {group}, iteration {step}', '--output', out / 'actual_vs_reference.mp4'],
                            out / 'render.log', state, state_path, 'rendering',
                            dict(env, PATH='/usr/bin:' + env.get('PATH', '')), timeout=1800)
                state['video_complete'] = True
                save_json(state_path, state)
            state['upload_skipped'] = 'user_requested_local_only'
            state.update(phase='complete', updated=now())
            save_json(state_path, state)
            remaining.remove((group, step))
            progressed = True
        report(cfg, root, jobs)
        if remaining and not progressed:
            save_json(root / 'waiting.json', {'updated': now(), 'pending': remaining,
                      'free_mib': free_memory(cfg.get('gpu', 0)), 'min_free_mib': args.min_free_mib})
            print(f'{now()} Waiting for checkpoints/GPU memory: {remaining}', flush=True)
            time.sleep(30)


if __name__ == '__main__':
    main()
