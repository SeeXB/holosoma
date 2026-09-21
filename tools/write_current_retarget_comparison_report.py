#!/usr/bin/env python3
"""Summarize only completed, source-matched Original/B4 pairs; expose failed runs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from run_current_semantic_retarget_comparison import ROOT, METRICS, write


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison')
    args=ap.parse_args(); output=args.output.resolve()
    manifest=json.loads((output/'manifest.json').read_text())
    rows=[]; pairs=[]; mismatched=[]
    for job in manifest['jobs']:
        p=Path(job['run'])/'metrics.json'
        row=json.loads(p.read_text()) if p.exists() else {'status':'pending'}
        if row.get('signature') and row['signature']!=job['signature']:row={'status':'stale'}
        row = {**{k:job[k] for k in ('dataset','task','method','run')},**row}
        if row['status']=='ok' and row['dataset']=='omomo':
            audit=Path(job['run'])/'profile/penetration_pairs.csv'
            with audit.open() as stream:
                contacts=list(csv.DictReader(stream))
            row['max_hand_object_penetration_mm']=1000*max((float(c['penetration_depth']) for c in contacts if c['pair_type']=='hand-box'),default=0.)
            row['max_all_body_object_penetration_mm']=1000*max((float(c['penetration_depth']) for c in contacts if c['pair_type'] in {'hand-box','other-body-box'}),default=0.)
        rows.append(row)
    by={(r['dataset'],r['task'],r['method']):r for r in rows}
    for ds,task in dict.fromkeys((j['dataset'],j['task']) for j in manifest['jobs']):
        a,b=[by[(ds,task,m)] for m in ('original','semantic_b4')]
        if a['status']!='ok' or b['status']!='ok':continue
        if any(a.get(k)!=b.get(k) or not a.get(k) for k in ('source_vertices_sha256','source_adjacency_sha256')):
            mismatched.append({'dataset':ds,'task':task});continue
        pairs.append({'dataset':ds,'task':task,'original':a,'semantic_b4':b})
    aggs=[]
    for ds in ('omomo','lafan'):
        part=[p for p in pairs if p['dataset']==ds]
        for metric in [*METRICS,'max_hand_object_penetration_mm','max_all_body_object_penetration_mm']:
            valid=[p for p in part if p['original'].get(metric) is not None and p['semantic_b4'].get(metric) is not None]
            if not valid:continue
            a=float(np.mean([p['original'][metric] for p in valid]));b=float(np.mean([p['semantic_b4'][metric] for p in valid]))
            aggs.append(dict(dataset=ds,metric=metric,n_pairs=len(valid),original=a,semantic_b4=b,relative_delta_percent=100*(b/a-1) if a else None))
    payload={'requested_runs':len(rows),'status_counts':{s:sum(r['status']==s for r in rows) for s in ('ok','failed','pending','stale')},'matched_pairs':len(pairs),'source_mismatches':mismatched,'aggregates':aggs,'rows':rows}
    payload['metric_conventions'] = {
        'aggregation': 'Equal-weight task means over source-matched completed pairs, separately per dataset.',
        'penetration_fraction': 'Fraction of all frames with a considered penetration deeper than 10 mm.',
        'sliding_fraction': 'Fraction of reference foot-contact frames with detected sliding.',
        'mean_positive_penetration_mm': 'Mean frame maximum over frames exceeding the official threshold, zero for no such frames; then equal-weight task mean.',
        'toe_sliding_velocity_m_s': 'Legacy field name: the evaluator actually returns displacement in metres per frame, without division by dt; zero when no sliding frames.',
        'max_body_box_penetration_mm': 'Legacy field excludes hand-box contacts; correctly labeled non-hand body-object in the table.',
        'max_all_body_object_penetration_mm': 'Maximum over both hand-box and other-body-box contacts, per task; aggregated as an equal-weight task mean.',
    }
    write(output/'evaluation.json',payload)
    fmt=lambda x: '—' if x is None else f'{x:.3f}'
    lines=['# 新 Semantic Plan：Semantic B4 vs Original OmniRetarget','',
           f"运行状态：{payload['status_counts']}；完整且源几何一致的配对任务 **{len(pairs)}**。",'',
           '## 统一协议','',
           '- 两组重新运行，同任务输入轨迹、场景、机器人及随机采样一致；输入与评测计划均记录哈希。两组结果的源交互顶点及邻接矩阵哈希一致才纳入配对统计。',
           '- 使用新版计划的相同事件帧/关键部位评估两组。精度基于未加权Laplacian残差及原始交互网格，不能把加权优化cost当作可比误差。',
           '- B4保留Uniform-2、全事件空间加权、transition truncation、exact-trigger budget=4；hand维持4倍，新增解剖部位显式2倍，物体邻点2倍。Original不使用语义加权。hand与wrist在当前交互网格可映射到同一顶点，并不是两个独立手腕/手掌几何测量。',
           '- 保留关节限制、防穿透与foot sticking；容差1mm，step=0.2。大桌使用已验证的共享compound场景；其余OMOMO使用official_inputs场景。不关闭约束或修改容差来掩盖不可行。',
           '- LAFAN使用完整原生帧（stride=1）和真正的Semantic B4；不使用旧stride-20/Uniform-2替代。LAFAN不报告人体—物体边误差。',
           '- 精度与穿透深度单位mm；penetration_fraction以全部帧为分母，sliding_fraction以参考足部接触帧为分母，均不是秒。官方穿透评估忽略小于等于10mm的穿透，因此另报更敏感的几何审计最大深度；0帧占比不等于零几何穿透。',
           '- mean_positive_penetration_mm先取每任务超阈值帧的最大深度均值，无此类帧记0，再对任务等权平均。原始字段toe_sliding_velocity_m_s沿用旧命名，但现有官方实现没有除以dt，实际单位是m/frame；表中按真实位移单位标注，不当作m/s。',
           '- 穿透口径更正：旧字段max_body_box_penetration_mm只统计other-body-box，排除了hand-box，不能称为全部人体—物体穿透。表中改标non_hand并补充手部及包含手部的全身最大深度；expected_contact标签不代表允许任意深度穿透。',
           '- CPU并发运行，耗时受系统负载影响；SQP/solver calls较稳定。失败任务单列，均值仅来自两组均完成的相同任务，不能代表全部37任务。接触preservation存在饱和可能，不单凭该指标判断质量。',
           '- 未启动RL训练，结果仅本地保存。','']
    improved = output.parent / 'g1_anatomy_v2_20260920_geometry_benchmark/BENCHMARK_REPORT.md'
    if improved.exists():
        lines += ['## 后续穿透优化实验', '', '新增B4+几何修正的完整轨迹计时与指标见 [独立对照报告](../g1_anatomy_v2_20260920_geometry_benchmark/BENCHMARK_REPORT.md)。以下表格仍是原B4基线，不混入改进版结果。', '']
    for ds in ('omomo','lafan'):
        lines += [f'## {ds.upper()} 配对均值','', '| 指标 | Original | Semantic B4 | 相对变化 | 配对数 |','|---|---:|---:|---:|---:|']
        for r in aggs:
            if r['dataset']==ds:
                name = 'toe_sliding_displacement_m_per_frame' if r['metric']=='toe_sliding_velocity_m_s' else r['metric']
                if name=='max_body_box_penetration_mm':name='max_non_hand_body_object_penetration_mm'
                delta = '—' if r['relative_delta_percent'] is None else f"{r['relative_delta_percent']:+.3f}%"
                lines.append(f"| {name} | {fmt(r['original'])} | {fmt(r['semantic_b4'])} | {delta} | {r['n_pairs']} |")
    lines += ['', '## 每任务结果', '', '| 任务 | Original | B4 | Original Part Exact mm | B4 Part Exact mm |', '|---|---|---|---:|---:|']
    for ds,task in dict.fromkeys((j['dataset'],j['task']) for j in manifest['jobs']):
        a,b=[by[(ds,task,m)] for m in ('original','semantic_b4')]
        lines.append(f"| {ds}/{task} | {a['status']} | {b['status']} | {fmt(a.get('semantic_part_exact_mm'))} | {fmt(b.get('semantic_part_exact_mm'))} |")
    lines += ['', '## 失败原因', '']
    for r in rows:
        if r['status']=='failed':lines.append(f"- **{r['dataset']}/{r['task']}/{r['method']}**：{r.get('error','unknown').splitlines()[-1]}。日志：`{r['run']}/retarget.log`。")
    (output/'EVALUATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:payload[k] for k in ('requested_runs','status_counts','matched_pairs','source_mismatches')}))


if __name__=='__main__':main()
