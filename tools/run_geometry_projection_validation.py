#!/usr/bin/env python3
"""Validate remaining previously completed B4 tasks after the serial benchmark."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from run_current_semantic_retarget_comparison import ROOT,write
from run_geometry_projection_benchmark import report

old=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison'
benchmark=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_benchmark'
out=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_validation'
out.mkdir(parents=True,exist_ok=True)
lock=(out/'validation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (benchmark/'COMPLETE.json').exists():time.sleep(15)
fingerprint=json.loads((benchmark/'manifest.json').read_text())['code_sha256']
for path,digest in fingerprint.items():
    if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:raise RuntimeError('Code changed after timing benchmark')
jobs=[]
for task in json.loads((benchmark/'remaining_validation_tasks.json').read_text()):
    job=json.loads((old/'omomo'/task/'semantic_b4/job.json').read_text());run=out/task/'semantic_b4_projection'/'repeat_0'
    job.update(method='semantic_b4_projection',run=str(run),repeat=0,geometry_projection=True,code_sha256=fingerprint)
    job['command']=[sys.executable,str(ROOT/'tools/run_current_semantic_retarget_comparison.py'),'--execute-retarget',str(run/'job.json')]
    job['signature']=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest();write(run/'job.json',job);jobs.append(job)
write(out/'manifest.json',{'jobs':jobs,'workers':2,'note':'Quality validation only; parallel runtimes are not used for serial speed claims.','code_sha256':fingerprint})
env=dict(os.environ,PYTHONPATH=str(ROOT/'src/holosoma_retargeting')+':'+str(ROOT/'tools'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def quality_report():
    report(out,jobs)
    p=out/'BENCHMARK_REPORT.md'
    text=p.read_text().replace('B4几何修正：完整轨迹精度与串行计时','B4几何修正：扩展质量验证').replace('同机单任务串行、CPU线程数1，方法顺序轮换。重复运行均单独保存。','两任务并发、每任务CPU线程数1；本文件耗时不得用于串行速度结论。')
    p.write_text(text)

def run(job):
    d=Path(job['run'])
    with (d/'worker.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/run_current_semantic_retarget_comparison.py'),'--worker',str(d/'job.json')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    row=json.loads((d/'metrics.json').read_text());print(job['task'],row['status'],row.get('error','').splitlines()[-1:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    pending=[pool.submit(run,job) for job in jobs]
    for future in concurrent.futures.as_completed(pending):future.result();quality_report()
write(out/'COMPLETE.json',{'attempted':len(jobs),'serial_timing':False})
