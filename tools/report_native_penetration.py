#!/usr/bin/env python3
"""Recompute penetration arrays and use the native evaluator's aggregation."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.evaluation.eval_retargeting import RetargetingEvaluator, create_task_constants
from prepare_batch_retarget_inputs import TASK_OBJECTS

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'exp/retargeting/g1_anatomy_v2_20260920_bilateral_no_projection'
PACKAGE = ROOT / 'src/holosoma_retargeting/holosoma_retargeting'


def aggregate(rows, method):
    duration = np.asarray([r[method]['duration'] for r in rows])
    depths = np.asarray([v * 100 for r in rows for v in r[method]['penetration_max_depths_m']])
    return dict(tasks=len(rows), duration_mean=float(duration.mean()), duration_std=float(duration.std()),
                penetrating_frames=len(depths),
                max_depth_cm_mean=float(depths.mean()) if len(depths) else 0.0,
                max_depth_cm_std=float(depths.std()) if len(depths) else 0.0)


def main():
    source = RUN / 'bilateral_evaluation.json'
    comparison = json.loads(source.read_text())
    exclusion = json.loads((RUN / 'excluded_tasks_analysis.json').read_text())['excluded_tasks']
    rows = []
    os.chdir(PACKAGE)
    for pair in comparison['rows']:
        if pair['current']['status'] != 'ok' or pair['previous']['original']['status'] != 'ok':
            continue
        task = pair['task']
        job = json.loads((RUN / 'omomo' / task / 'semantic_b4/job.json').read_text())
        obj = TASK_OBJECTS[task]
        constants = create_task_constants(RobotConfig(robot_type='g1'),
                                         MotionDataConfig(data_format='smplh', robot_type='g1'), object_name=obj)
        constants.SCENE_XML_FILE = job['scene']
        evaluator = RetargetingEvaluator(constants.ROBOT_URDF_FILE, getattr(constants, 'OBJECT_URDF_FILE', None),
                                        obj, constants.DEMO_JOINTS, constants.JOINTS_MAPPING,
                                        visualize=False, constants=constants)
        row = dict(task=task, excluded=task in exclusion, scene=job['scene'])
        for method, metrics in [('original', pair['previous']['original']), ('bilateral_b4', pair['current'])]:
            path = Path(metrics['result'])
            with np.load(path, allow_pickle=False) as data:
                qpos = np.asarray(data['qpos'])
            duration, depths = evaluator.evaluate_penetration(qpos)
            # Verify against already published per-task evaluator outputs.
            assert np.isclose(duration, metrics['penetration_fraction'], atol=1e-12)
            assert np.isclose(1000*np.mean(depths) if depths else 0, metrics['mean_positive_penetration_mm'], atol=1e-6)
            row[method] = dict(trajectory=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               frames=len(qpos), duration=duration, penetration_max_depths_m=depths)
        rows.append(row)
        (RUN / 'native_penetration_raw.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(f'Validated {task}', flush=True)
    subsets = {'all_paired': rows, 'user_excluded_subset': [r for r in rows if not r['excluded']]}
    summary = {key: {method: aggregate(subset, method) for method in ('original', 'bilateral_b4')}
               for key, subset in subsets.items()}
    report = dict(excluded_tasks=exclusion, threshold_m=0.01, std_ddof=0,
                  duration_aggregation='mean/std of per-trajectory fractions',
                  depth_aggregation='concatenate per-penetrating-frame maximum depths; mean/std; no zero padding',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), summary=summary)
    (RUN / 'native_penetration_summary.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# 原生穿透指标：Duration 与 Max Depth', '',
             '按原生evaluate_penetration及main的汇总逻辑重算，包含手部；超过10mm的帧才进入深度列表。'
             'Duration先逐任务计算帧比例，再对任务取mean/std。Max Depth对每个超阈值帧取最大深度，'
             '跨任务拼接这些深度后取mean/std（cm），不把无穿透任务补零。标准差ddof=0。', '',
             '这与此前的“每任务最大穿透均值”和“每任务超阈值深度均值、无穿透补零”均不同。', '',
             '论文定义参考：https://arxiv.org/html/2509.26633v1#S5.SS2 。论文完整数据集与本地任务子集不同，不能直接用本文数值复现论文表格。', '',
             '用户排除：'+', '.join(exclusion)+'.', '']
    for name, values in summary.items():
        lines += [f'## {name}（{values["original"]["tasks"]} 对）', '',
                  '|方法|Duration↓（比例）|Max Depth↓（cm）|超阈值帧数|', '|---|---:|---:|---:|']
        for method, m in values.items():
            lines.append(f"|{method}|{m['duration_mean']:.6f} ± {m['duration_std']:.6f}|"
                         f"{m['max_depth_cm_mean']:.4f} ± {m['max_depth_cm_std']:.4f}|{m['penetrating_frames']}|")
    (RUN / 'NATIVE_PENETRATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
