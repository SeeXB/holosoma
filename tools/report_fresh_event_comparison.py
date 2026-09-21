#!/usr/bin/env python3
"""Report fresh event recognition and source-matched Original/B4 results."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--generation',type=Path,required=True)
    args=parser.parse_args()
    generation=json.loads(args.generation.read_text())
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'comparison.json'
    comparison=json.loads(path.read_text()) if path.exists() else dict(rows=[],paired=[],finished_runs=0,requested_runs=0)
    pairs=comparison['paired']
    summary={}
    lines=['# Fresh trajectory-event plans: Semantic B4 vs Original OmniRetarget','',
           f"Generation: {generation['finished']}/{generation['total']} attempts finished. Retargeting: "
           f"{comparison['finished_runs']}/{comparison['requested_runs']} runs; {len(pairs)} matched pairs.", '',
           'Fresh whole-clip VLM requests only; event intervals come from trajectory functions. '
           'Failed/untriggered plans are not replaced with older plans. '
           'B4 geometry projection remains disabled; original physical constraints remain enabled. '
           'Results are local; no RL training or W&B upload.', '',
           'Duration is the per-trajectory fraction of frames with penetration >10 mm, averaged over tasks. '
           'Max Depth is mean/std of concatenated per-penetrating-frame maximum depths (cm), including hands; '
           'no zero-padding of depth samples from non-penetrating tasks. '
           'Timing is full retargeting subprocess elapsed time, including startup, loading, collision checks and saving, '
           'excluding the subsequent evaluation. Retargeting jobs run serially.', '']
    metrics=['semantic_part_exact_mm','global_exact_mm','local_exact_mm','body_object_edge_exact_mm',
             'ordinary_mm','contact_preservation','sliding_fraction','end_to_end_retarget_time_s','sqp_iterations']
    for ds in ('omomo','lafan'):
        selected=[p for p in pairs if p['dataset']==ds]
        summary[ds]=dict(pairs=len(selected),metrics={},penetration={})
        lines += [f'## {ds}: {len(selected)} matched pairs','',
                  '|Metric|Original|Semantic B4|Change|','|---|---:|---:|---:|']
        for metric in metrics:
            valid=[p for p in selected if all(p[m].get(metric) is not None for m in ('original','semantic_b4'))]
            if not valid:continue
            a,b=[float(np.mean([p[m][metric] for p in valid])) for m in ('original','semantic_b4')]
            delta=100*(b/a-1) if a else None
            summary[ds]['metrics'][metric]=dict(original=a,semantic_b4=b,relative_percent=delta)
            change=f'{delta:+.2f}%' if delta is not None else '—'
            lines.append(f'|{metric}|{a:.5f}|{b:.5f}|{change}|')
        for method in ('original','semantic_b4'):
            samples=[json.loads((Path(p[method]['result']).parent/'native_penetration_raw.json').read_text()) for p in selected]
            duration=[s['duration'] for s in samples]
            depth=[v*100 for s in samples for v in s['penetration_max_depths_m']]
            if not samples:continue
            values=dict(duration_mean=float(np.mean(duration)),duration_std=float(np.std(duration)),
                        max_depth_cm_mean=float(np.mean(depth)) if depth else 0,
                        max_depth_cm_std=float(np.std(depth)) if depth else 0,penetrating_frames=len(depth))
            summary[ds]['penetration'][method]=values
            lines.append(f"|{method}: Duration|{values['duration_mean']:.6f} ± {values['duration_std']:.6f}|||" )
            lines.append(f"|{method}: Max Depth (cm)|{values['max_depth_cm_mean']:.4f} ± {values['max_depth_cm_std']:.4f}|||" )
        lines += ['']
    lines += ['## Generation status (all requested tasks)','', '|Task|Status|Detail|','|---|---|---|']
    for row in generation['results']:
        detail=row.get('error',row.get('error_type',''))
        if row.get('coverage'):detail=f"key frames {row['coverage']['key_frames']}/{row['coverage']['frames']}"
        lines.append(f"|{row['dataset']}/{row['task']}|{row['status']}|{detail}|")
    lines += ['', '## Retargeting failures', '']
    for row in comparison['rows']:
        if row['status']!='ok':lines.append(f"- {row['dataset']}/{row['task']}/{row['method']}: {row.get('error','unknown')}")
    if not pairs:lines += ['', '**No matched results are available; no improvement or degradation can be claimed.**']
    (args.output/'EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    (args.output/'evaluation.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()
