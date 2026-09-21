#!/usr/bin/env python3
"""Render complete bilateral B4 trajectories, with selected synchronized comparisons."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')

import mujoco
import numpy as np
from PIL import Image

from holosoma_retargeting.examples.render_mujoco_trajectory_comparison import (
    CameraConfig, RawVideoEncoder, _annotate, _camera_for_frame, _load_trajectory,
    _set_qpos, _sha256,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'exp/retargeting/g1_anatomy_v2_20260920_bilateral_no_projection'
OLD = ROOT / 'exp/retargeting/g1_anatomy_v2_20260920_comparison'
COMPARE = ('sub1_largetable_028', 'sub15_woodchair_020', 'sub9_monitor_025')


def render(job):
    task = job['task']
    run = Path(job['run'])
    result = json.loads((run / 'metrics.json').read_text())
    if result['status'] != 'ok':
        return dict(task=task, status='skipped_failed_retarget')
    output = RUN / 'videos' / task
    output.mkdir(parents=True, exist_ok=True)
    path = Path(result['result'])
    qpos, fps = _load_trajectory(path)
    old_run = OLD / 'omomo' / task / 'original'
    old_path = old_run / path.name
    original_metrics = json.loads((old_run / 'metrics.json').read_text())
    old_qpos = None
    if original_metrics['status'] == 'ok':
        old_qpos, old_fps = _load_trajectory(old_path)
        if old_fps != fps or old_qpos.shape != qpos.shape:
            raise ValueError(f'{task}: comparison trajectories are not aligned')
        if not np.array_equal(old_qpos[:, -7:], qpos[:, -7:]):
            raise ValueError(f'{task}: object motion mismatch')
    model = mujoco.MjModel.from_xml_path(job['scene'])
    width, height = 640, 480
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    data = mujoco.MjData(model)
    old_data = mujoco.MjData(model)
    config = CameraConfig(distance=3.8 if 'largetable' in task else 3.15)
    peaks = list(csv.DictReader((run / 'profile/penetration_pairs.csv').open()))
    peaks = [r for r in peaks if r['pair_type'] in ('hand-box', 'other-body-box')]
    worst = max(peaks, key=lambda r: float(r['penetration_depth'])) if peaks else None
    preview_indices = {0, len(qpos)//3, 2*len(qpos)//3, len(qpos)-1}
    if worst:
        preview_indices.add(int(worst['frame']))
    frames = []
    encoder = RawVideoEncoder(output / 'bilateral_b4.mp4', width=width, height=height, fps=fps)
    comparison = RawVideoEncoder(output / 'original_vs_bilateral_b4.mp4', width=2*width, height=height, fps=fps) if old_qpos is not None else None
    try:
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            for i, q in enumerate(qpos):
                previous = old_qpos[i] if old_qpos is not None else q
                camera = _camera_for_frame(previous, q, config)
                _set_qpos(model, data, q)
                renderer.update_scene(data, camera=camera)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                current = _annotate(renderer.render().copy(), title='Bilateral B4 | projection OFF',
                                    frame_index=i, frame_count=len(qpos), fps=fps, accent=(76,192,135))
                encoder.write(current)
                if comparison:
                    _set_qpos(model, old_data, previous)
                    renderer.update_scene(old_data, camera=camera)
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
                    before = _annotate(renderer.render().copy(), title='Original OmniRetarget',
                                       frame_index=i, frame_count=len(qpos), fps=fps, accent=(226,92,88))
                    panel = np.concatenate([before, current], axis=1)
                    comparison.write(panel)
                else:
                    panel = current
                if i in preview_indices:
                    frames.append(panel)
                if i % 60 == 0:
                    print(f"{task}: rendered {i + 1}/{len(qpos)}", flush=True)
    finally:
        encoder.close()
        if comparison:
            comparison.close()
    Image.fromarray(np.concatenate(frames, axis=0)).save(output / 'preview.jpg', quality=90)
    metadata = dict(task=task, status='ok', frames=len(qpos), fps=fps, duration_s=len(qpos)/fps,
                    trajectory=str(path), trajectory_sha256=_sha256(path), scene=job['scene'],
                    scene_sha256=_sha256(Path(job['scene'])), geometry_projection=False,
                    playback='Exact saved qpos; mj_forward only; no simulation or smoothing',
                    rendering={'shadows': False, 'reflections': False}, camera=config.__dict__, video=str(output/'bilateral_b4.mp4'),
                    comparison=str(output/'original_vs_bilateral_b4.mp4') if comparison else None,
                    original_trajectory=str(old_path) if comparison else None,
                    original_sha256=_sha256(old_path) if comparison else None,
                    max_penetration_audit_row=worst)
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main():
    jobs = json.loads((RUN/'manifest.json').read_text())['jobs']
    jobs.sort(key=lambda j: (COMPARE.index(j['task']) if j['task'] in COMPARE else len(COMPARE)))
    directory = RUN/'videos'
    directory.mkdir(exist_ok=True)
    rows = []
    for job in jobs:
        rows.append(render(job))
        (directory/'manifest.json').write_text(json.dumps(rows, indent=2)+'\n')
        lines = ['# 双侧B4轨迹视频', '', '全部为保存qpos的逐帧播放，无RL执行、平滑或物理仿真。新增穿透修正关闭。', '',
                 '对比视频：左为Original OmniRetarget，右为补齐左右后的B4；两侧使用同一相机及对应帧。', '',
                 '|任务|双侧B4|Original / 双侧B4|', '|---|---|---|']
        for r in rows:
            task = r['task']
            if r['status'] != 'ok':
                lines.append(f'|{task}|重定向失败，无完整轨迹|—|')
            else:
                link = f'[观看]({task}/original_vs_bilateral_b4.mp4)' if r['comparison'] else '—'
                lines.append(f'|{task}|[观看]({task}/bilateral_b4.mp4)|{link}|')
        (directory/'README.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps(dict(task=job['task'], status=rows[-1]['status'])), flush=True)


if __name__ == '__main__':
    main()
