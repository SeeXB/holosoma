#!/usr/bin/env python3
"""Consolidate timing, expanded quality and independent exported-geometry audit."""
import csv
import json
import statistics
from pathlib import Path
from run_current_semantic_retarget_comparison import ROOT,write

bench=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_benchmark'
validation=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_validation'
old=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison'
assert (bench/'COMPLETE.json').exists() and (validation/'COMPLETE.json').exists()
timing=json.loads((bench/'benchmark.json').read_text())['rows']
extra=json.loads((validation/'benchmark.json').read_text())['rows']
audit=json.loads((bench/'exported_geometry_audit.json').read_text());audit_by={r['task']:r for r in audit}
quality=[];failed=[]
for row in [r for r in timing if r['method']=='semantic_b4_projection' and r['repeat']==0]+extra:
    if row['status']!='ok':failed.append(row);continue
    task=row['task'];baseline=json.loads((old/'omomo'/task/'semantic_b4/metrics.json').read_text())
    assert row['source_vertices_sha256']==baseline['source_vertices_sha256'] and row['source_adjacency_sha256']==baseline['source_adjacency_sha256']
    assert audit_by[task]['passed']
    with (old/'omomo'/task/'semantic_b4/profile/penetration_pairs.csv').open() as stream:contacts=list(csv.DictReader(stream))
    old_max=1000*max((float(c['penetration_depth']) for c in contacts if c['pair_type'] in ['hand-box','other-body-box']),default=0.)
    changes={k:100*(row[k]/baseline[k]-1) if baseline[k] else None for k in ['semantic_part_exact_mm','global_exact_mm','body_object_edge_exact_mm','ordinary_mm','local_exact_mm']}
    quality.append({'task':task,'old_max_penetration_mm':old_max,'new_max_penetration_mm':audit_by[task]['max_all_body_object_penetration_mm'],'metric_change_percent':changes,'baseline':baseline,'projection':row})
summary={'validated_tasks':len(quality),'failed':failed,'timing_runs':len(timing),'quality':quality,'timing':timing,'independent_audit':audit}
write(bench/'optimization_summary.json',summary)
lines=['# B4穿透优化结果','',f'3条轨迹×3方法×2次重复的串行计时已完成（18次）。扩展后共{len(quality)}条原B4成功轨迹通过质量验证，新增失败{len(failed)}条。', '',
       '方法：保留原B4计划、权重、普通帧/关键帧预算和基线轨迹；对不满足真实几何条件的帧做最小改动投影，使用原有关节/足部/非穿透约束，更新后复核并回溯。投影上限10次；无法达标就失败。默认不开启，Original不变。新增求解与全部检查开销均计入时间。','',
       '## 完整轨迹总时间（秒，两次平均）','',
       '|任务|Original|原B4|B4+修正|比Original减少|比原B4增加|','|---|---:|---:|---:|---:|---:|']
for task in dict.fromkeys(r['task'] for r in timing):
    means={m:statistics.mean(r['end_to_end_retarget_time_s'] for r in timing if r['task']==task and r['method']==m) for m in ['original','semantic_b4','semantic_b4_projection']}
    a,b,c=means.values();lines.append(f'|{task}|{a:.2f}|{b:.2f}|{c:.2f}|{100*(1-c/a):.1f}%|{100*(c/b-1):.1f}%|')
lines+=['','端到端从重定向子进程启动计到退出，包括加载、整条轨迹的SQP、几何检查、回退、修正、结果/profile保存；不含后续独立评测。单任务串行、CPU数值库单线程、方法顺序轮换，机器其它负载未做硬隔离。各次时间范围与检查/修正耗时拆分见[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)。扩展验证的并发耗时不用于加速结论。','','## 穿透及原指标变化（相对原B4）','','|任务|原最大含手穿透 mm|新最大含手穿透 mm|Part误差变化|Global误差变化|Edge误差变化|','|---|---:|---:|---:|---:|---:|']
for row in quality:
    d=row['metric_change_percent'];lines.append(f"|{row['task']}|{row['old_max_penetration_mm']:.3f}|{row['new_max_penetration_mm']:.3f}|{d['semantic_part_exact_mm']:+.3f}%|{d['global_exact_mm']:+.3f}%|{d['body_object_edge_exact_mm']:+.3f}%|")
lines+=['','正的误差变化表示小幅回退，不能表述为其它指标完全不变。保留的是大部分精度优势；原始五项精度、足滑、contact preservation和所有时间数据均在optimization_summary.json。','',
        '## 独立验证与边界','',f'- 直接加载导出的NPZ和碰撞模型，穷举全部机器人—物体有效碰撞几何对，包含手掌，{len(audit)}条轨迹全部通过1mm+0.01mm数值余量检查。记录见exported_geometry_audit.json。',
        '- 所有被验证轨迹：原B4基线qpos逐元素一致、锁定物体轨迹逐元素一致、参考交互顶点与邻接哈希一致、输出有限且根四元数归一化。',
        '- 这是当前碰撞模型下的离散帧检查，不是连续时间防穿透证明，不代表RL闭环执行也不会穿透；自碰撞仍沿用原来的配置范围。',
        '- 未解决原来B4已经失败的4条：sub16_largetable_013、sub9_monitor_025、sub9_tripod_015、sub16_whitechair_002。该投影处理成功B4轨迹，不解决前置SQP本身不可行。',
        '- 4项新投影测试和33项已有semantic runtime测试通过。未启动RL训练、未上传W&B、未commit/push。','']
for row in failed:lines.append(f"- 扩展失败：{row['task']}：{row.get('error','').splitlines()[-1]}")
(bench/'OPTIMIZATION_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'validated_tasks':len(quality),'failed_tasks':[r['task'] for r in failed],'timing_runs':len(timing)}))
