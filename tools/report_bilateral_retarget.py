#!/usr/bin/env python3
"""Evaluate saved baselines with the bilateral plan; never modify baseline runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from holosoma_retargeting.semantic_keyframes.precision import evaluate_precision, load_precision_payload
from holosoma_retargeting.semantic_keyframes.runtime import load_semantic_plan_projection
from run_current_semantic_retarget_comparison import ROOT, write

OUTPUT = ROOT / 'exp/retargeting/g1_anatomy_v2_20260920_bilateral_no_projection'
BASELINE = ROOT / 'exp/retargeting/g1_anatomy_v2_20260920_comparison'


def penetration(run):
    path = run / 'profile/penetration_pairs.csv'
    if not path.exists():
        return None
    with path.open() as stream:
        depths = [float(row['penetration_depth']) * 1000 for row in csv.DictReader(stream)
                  if row.get('pair_type') in ('hand-box', 'other-body-box')]
    return max(depths, default=0.0)


def baseline(job, method):
    run = BASELINE / job['dataset'] / job['task'] / method
    row = json.loads((run / 'metrics.json').read_text())
    old_job = json.loads((run / 'job.json').read_text())
    # Plans change intentionally, but motion and collision inputs must match.
    for path, sha in job['sources'].items():
        if path != job['plan'] and old_job['sources'].get(path) != sha:
            raise ValueError(f'Baseline input mismatch: {path}')
    if row['status'] != 'ok':
        return row
    payload = load_precision_payload(Path(row['result']))
    result = evaluate_precision(payload, load_semantic_plan_projection(job['plan']), critical_event_names=None)
    row.update(semantic_part_exact_mm=1000 * result.semantic_part['exact'],
               global_exact_mm=1000 * result.keyframe_global['exact'],
               local_exact_mm=1000 * result.semantic_local['exact'],
               body_object_edge_exact_mm=1000 * result.semantic_edge['exact'],
               ordinary_mm=1000 * result.ordinary['mean'],
               max_all_body_object_penetration_mm=penetration(run),
               reevaluated_plan=job['plan'], reused_trajectory=True)
    return row


def fmt(value):
    return '—' if value is None else f'{value:.3f}'


def main():
    jobs = json.loads((OUTPUT / 'manifest.json').read_text())['jobs']
    rows = []
    for job in jobs:
        run = Path(job['run'])
        path = run / 'metrics.json'
        if not path.exists():
            continue
        if job.get('geometry_projection') is not False:
            raise ValueError('This experiment must disable the added geometry projection')
        current = json.loads(path.read_text())
        if current['status'] == 'ok':
            current['max_all_body_object_penetration_mm'] = penetration(run)
        previous = {method: baseline(job, method) for method in ('original', 'semantic_b4')}
        for old in previous.values():
            if old['status'] == current['status'] == 'ok':
                for field in ('source_vertices_sha256', 'source_adjacency_sha256'):
                    if current[field] != old[field]:
                        raise ValueError(f'Source geometry mismatch: {job["task"]}/{field}')
        rows.append(dict(task=job['task'], current=current, previous=previous))
    ok = sum(r['current']['status'] == 'ok' for r in rows)
    summary = dict(attempted=len(rows), total=len(jobs), successful=ok, failed=len(rows)-ok,
                   geometry_projection=False, original_nonpenetration_constraints=True,
                   baseline_precision_reevaluated_with_current_plan=True, rows=rows)
    write(OUTPUT / 'bilateral_evaluation.json', summary)
    lines = ['# 双侧补齐后的 B4 重定向', '',
             f'已尝试 {len(rows)}/{len(jobs)}；成功 {ok}，失败 {len(rows)-ok}。', '',
             '本轮按用户确认的双手任务，对18个未完成OMOMO的所有已标注上肢区域补齐左右。'
             '只改body_parts，不改动作阶段/触发帧、输入轨迹、物体几何、权重倍率与迭代预算。', '',
             '**关闭的是新增几何投影修正（geometry_projection=false）；原始B4非穿透、足部、关节约束保留。**', '',
             '旧B4与Original轨迹复用，精度指标使用本轮同一双侧plan重新计算，避免评价部位变化造成假差异。'
             '穿透含hand-box与other-body-box，为原求解器几何审计值。原始视觉响应保留为证据，'
             '其中单侧contact_review不代表最终body_parts仍是单侧。', '',
             '本轮总时间从重定向子进程启动计到退出，包含加载、求解、几何审计和保存，不含随后独立eval。'
             '任务串行、数值库单线程，每条一次；不与旧并发运行的时间直接推导加速比。', '',
             '|任务|旧B4|双侧B4|旧/新最大穿透 mm|旧/新Part mm|新总时间 s|',
             '|---|---|---|---:|---:|---:|']
    for row in rows:
        old, new = row['previous']['semantic_b4'], row['current']
        lines.append(f"|{row['task']}|{old['status']}|{new['status']}|"
                     f"{fmt(old.get('max_all_body_object_penetration_mm'))} / {fmt(new.get('max_all_body_object_penetration_mm'))}|"
                     f"{fmt(old.get('semantic_part_exact_mm'))} / {fmt(new.get('semantic_part_exact_mm'))}|"
                     f"{fmt(new.get('end_to_end_retarget_time_s'))}|")
    for method in ('semantic_b4', 'original'):
        pairs = [r for r in rows if r['current']['status'] == r['previous'][method]['status'] == 'ok']
        lines += ['', f'## 与 {method} 的共同成功任务（{len(pairs)} 对）', '',
                  '|指标|旧轨迹（按双侧plan评测）|双侧B4|相对变化|', '|---|---:|---:|---:|']
        for metric in ('semantic_part_exact_mm', 'global_exact_mm', 'body_object_edge_exact_mm',
                       'max_all_body_object_penetration_mm', 'sliding_fraction', 'contact_preservation'):
            if not pairs:
                continue
            a = float(np.mean([r['previous'][method][metric] for r in pairs]))
            b = float(np.mean([r['current'][metric] for r in pairs]))
            change = f'{100*(b/a-1):+.2f}%' if a else '—'
            lines.append(f'|{metric}|{a:.5f}|{b:.5f}|{change}|')
    lines += ['', '失败仍按失败计，不用部分轨迹补齐。未启动RL训练或上传W&B。', '']
    (OUTPUT / 'EVALUATION_REPORT.md').write_text('\n'.join(lines))
    if len(rows) == len(jobs):
        write(OUTPUT / 'COMPLETE.json', {k:v for k,v in summary.items() if k != 'rows'})
    print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
