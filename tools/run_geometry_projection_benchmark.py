#!/usr/bin/env python3
"""Serial full-trajectory quality and end-to-end timing comparison, local only."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import numpy as np
from run_current_semantic_retarget_comparison import ROOT, write

TASKS=['sub1_largetable_028','sub10_whitechair_118','sub12_tripod_041']
METHODS=['original','semantic_b4','semantic_b4_projection']
FILES=['tools/run_geometry_projection_benchmark.py','tools/run_current_semantic_retarget_comparison.py','src/holosoma_retargeting/holosoma_retargeting/src/interaction_mesh_retargeter.py','src/holosoma_retargeting/holosoma_retargeting/config_types/semantic.py','src/holosoma_retargeting/holosoma_retargeting/semantic_keyframes/geometry_projection.py']


def report(output, jobs):
    rows=[]
    for job in jobs:
        file=Path(job['run'])/'metrics.json'
        if not file.exists():continue
        row=json.loads(file.read_text());row['repeat']=job['repeat']
        if row['status']=='ok':
            with (file.parent/'profile/penetration_pairs.csv').open() as stream:contacts=list(csv.DictReader(stream))
            row['max_all_body_object_penetration_mm']=1000*max((float(x['penetration_depth']) for x in contacts if x['pair_type'] in {'hand-box','other-body-box'}),default=0.)
            frames=json.loads((file.parent/'profile/profile.json').read_text())['frames']
            for key in ['geometry_projection_time','geometry_projection_check_time','geometry_projection_solve_time','base_optimization_time']:
                row[key]=sum(x.get(key,0.) for x in frames)
            row['geometry_projection_iterations']=sum(x.get('geometry_projection_iterations',0) for x in frames)
        rows.append(row)
    write(output/'benchmark.json',{'requested':len(jobs),'finished':len(rows),'rows':rows})
    lines=['# B4几何修正：完整轨迹精度与串行计时','',f'完成 {len(rows)}/{len(jobs)}；同机单任务串行、CPU线程数1，方法顺序轮换。重复运行均单独保存。', '',
           '端到端时间覆盖子进程启动、输入加载、完整重定向、碰撞检查/回退/修正、轨迹与profile保存；不含随后独立的评测。内部loop时间另列，其历史实现扣除了纯诊断trace时间，因此速度结论以端到端为主。不是仅按SQP次数推断时间。', '',
           '修正保留B4权重与原基线求解路径，在每帧基线结果附近做最小几何调整；关节、足部、物体与地面防穿透均检查，手掌不豁免。物体位置固定；穿透容差1mm，数值余量0.01mm。未通过检查会失败，不输出成功轨迹。', '',
           '|任务/方法|完成重复数|端到端总时间均值 s|实测范围 s|内部loop s|修正总耗时 s|修正几何检查 s|Part Exact mm|Global Exact mm|Edge Exact mm|最大含手穿透 mm|总迭代数|',
           '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for task in dict.fromkeys(j['task'] for j in jobs):
        for method in dict.fromkeys(j['method'] for j in jobs if j['task']==task):
            rr=[r for r in rows if r['task']==task and r['method']==method and r['status']=='ok']
            if not rr:continue
            mean=lambda k:float(np.mean([r[k] for r in rr]))
            times=[r['end_to_end_retarget_time_s'] for r in rr]
            vals=[mean(k) for k in ['wall_time_s','geometry_projection_time','geometry_projection_check_time','semantic_part_exact_mm','global_exact_mm','body_object_edge_exact_mm','max_all_body_object_penetration_mm','sqp_iterations']]
            lines.append(f"|{task}/{method}|{len(rr)}|{mean('end_to_end_retarget_time_s'):.3f}|{min(times):.3f}–{max(times):.3f}|"+'|'.join(f'{v:.3f}' for v in vals)+'|')
    lines+=['','## 完整指标与失败','', 'Global/Part/Edge使用相同源轨迹、邻接、语义触发帧与未加权评测。完整Ordinary/Local误差、穿透占比、足滑、接触preservation、solver次数及分项耗时见benchmark.json。最大穿透是每轨迹所有帧的最大值，表中对重复运行取均值。','']
    for row in rows:
        if row['status']!='ok':lines.append(f"- {row['task']}/{row['method']}/repeat{row['repeat']}: {row.get('error','').splitlines()[-1]}")
    (output/'BENCHMARK_REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_benchmark')
    ap.add_argument('--tasks',nargs='+',default=TASKS)
    ap.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    ap.add_argument('--repeats',type=int,default=2)
    args=ap.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'benchmark.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison'
    fingerprint={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES}
    for name in FILES:
        target=output/'code_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    jobs=[]
    for repeat in range(args.repeats):
        for index,task in enumerate(args.tasks):
            shift=(repeat+index)%len(args.methods);methods=args.methods[shift:]+args.methods[:shift]
            for method in methods:
                source_method='original' if method=='original' else 'semantic_b4'
                job=json.loads((old/'omomo'/task/source_method/'job.json').read_text())
                run=output/task/method/f'repeat_{repeat}'
                job.update(method=method,run=str(run),repeat=repeat,geometry_projection=(method=='semantic_b4_projection'),code_sha256=fingerprint)
                if method=='original':
                    command=job['command'];command[0]=sys.executable
                    command[command.index('--save-dir')+1]=str(run);command[command.index('--semantic.profile-dir')+1]=str(run/'profile')
                else:command=[sys.executable,str(ROOT/'tools/run_current_semantic_retarget_comparison.py'),'--execute-retarget',str(run/'job.json')]
                job['command']=command;job['signature']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()
                write(run/'job.json',job);jobs.append(job)
    write(output/'manifest.json',{'jobs':jobs,'serial':True,'threads':1,'code_sha256':fingerprint})
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src/holosoma_retargeting')+':'+str(ROOT/'tools'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for job in jobs:
        for name,sha in fingerprint.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Benchmark code changed: {name}')
        d=Path(job['run']);file=d/'metrics.json'
        if file.exists() and json.loads(file.read_text()).get('signature')==job['signature']:continue
        write(output/'RUNNING.json',{'task':job['task'],'method':job['method'],'repeat':job['repeat'],'loadavg':os.getloadavg()})
        with (d/'worker.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/run_current_semantic_retarget_comparison.py'),'--worker',str(d/'job.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=json.loads(file.read_text());print(job['task'],job['method'],job['repeat'],row['status'],row.get('end_to_end_retarget_time_s'),flush=True)
        report(output,jobs)
    report(output,jobs);write(output/'COMPLETE.json',{'attempted':len(jobs)})


if __name__=='__main__':main()
