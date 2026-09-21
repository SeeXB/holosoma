#!/usr/bin/env python3
"""Independently audit exported poses with exhaustive robot-object geom pairs."""
import json
from pathlib import Path
import mujoco
import numpy as np
from run_current_semantic_retarget_comparison import ROOT,TASK_OBJECTS,write

bench=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_benchmark'
validation=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_geometry_validation'
old=ROOT/'exp/retargeting/g1_anatomy_v2_20260920_comparison'
rows=[]
files=list(bench.glob('*/semantic_b4_projection/repeat_0/metrics.json'))+list(validation.glob('*/semantic_b4_projection/repeat_0/metrics.json'))
for file in files:
    metric=json.loads(file.read_text())
    if metric['status']!='ok':continue
    job=json.loads((file.parent/'job.json').read_text());task=job['task'];obj=TASK_OBJECTS[task]
    baseline=json.loads((old/'omomo'/task/'semantic_b4/metrics.json').read_text())
    with np.load(baseline['result'],allow_pickle=False) as data:base=data['qpos']
    with np.load(metric['result'],allow_pickle=False) as data:q=data['qpos'];coarse=data['base_qpos']
    model=mujoco.MjModel.from_xml_path(job['scene']);data=mujoco.MjData(model)
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,i) or '' for i in range(model.ngeom)]
    collidable=[i for i in range(model.ngeom) if model.geom_contype[i] or model.geom_conaffinity[i]]
    objects=[i for i in collidable if obj in names[i]]
    robot=[i for i in collidable if i not in objects and 'ground' not in names[i]]
    pairs=[(a,b) for a in robot for b in objects]
    maximum=0.;worst=None;over=0;buf=np.zeros(6)
    for frame,pose in enumerate(q):
        data.qpos[:]=pose;mujoco.mj_forward(model,data);frame_max=0.
        for a,b in pairs:
            distance=float(mujoco.mj_geomDistance(model,data,a,b,.01,buf));depth=max(0.,-distance)
            frame_max=max(frame_max,depth)
            if depth>maximum:maximum=depth;worst={'frame':frame,'geoms':[names[a],names[b]]}
        over+=int(frame_max>.00101)
    row={'task':task,'frames':len(q),'pairs_per_frame':len(pairs),'queries':len(q)*len(pairs),'max_all_body_object_penetration_mm':maximum*1000,'frames_over_1_01mm':over,'worst':worst,'finite':bool(np.isfinite(q).all()),'base_qpos_max_abs_difference':float(np.max(np.abs(coarse-base))),'locked_object_max_abs_difference':float(np.max(np.abs(q[:,-7:]-base[:,-7:]))),'quaternion_norm_max_error':float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=1)-1))),'source_vertices_match':metric['source_vertices_sha256']==baseline['source_vertices_sha256'],'source_adjacency_match':metric['source_adjacency_sha256']==baseline['source_adjacency_sha256']}
    row['passed']=row['finite'] and row['base_qpos_max_abs_difference']==0 and row['locked_object_max_abs_difference']==0 and row['quaternion_norm_max_error']<1e-8 and over==0 and row['source_vertices_match'] and row['source_adjacency_match']
    rows.append(row);print(json.dumps(row),flush=True);write(bench/'exported_geometry_audit.json',rows)
if not rows or not all(row['passed'] for row in rows):raise SystemExit('Exported geometry validation failed')
