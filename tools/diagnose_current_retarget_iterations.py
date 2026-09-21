"""Replay selected B4 frames; probe iteration/weight effects without accepting probe poses."""
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path
import numpy as np
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--job',type=Path,required=True)
parser.add_argument('--frames',type=int,nargs='+',required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
job=json.loads(args.job.read_text());root=Path(__file__).resolve().parents[1]
with np.load(Path(job['run'])/(job['task']+'_original.npz'),allow_pickle=False) as data:reference=data['qpos']
args.output.mkdir(parents=True,exist_ok=True)
job['run']=str(args.output);replay=args.output/'job.json';replay.write_text(json.dumps(job,indent=2))
original=InteractionMeshRetargeter.iterate
records=[]
class FinishedProbe(Exception):pass

def iterate(self,**kw):
 frame=kw['frame_idx']
 if frame not in args.frames:return original(self,**kw)
 saved=copy.deepcopy(kw);baseline=original(self,**kw)
 row={'frame':frame,'baseline_budget':kw['n_iter'],'baseline_qpos_max_abs_difference':float(np.max(np.abs(baseline[0]-reference[frame]))),'probes':{}}
 for name,budget,unweighted in [('same_weights_10_iterations',10,False),('unweighted_same_budget',kw['n_iter'],True)]:
  probe=copy.deepcopy(saved);trace=[]
  def observe(i,q,cost):
   _,pairs=self._penetration_audit(q,frame)
   relevant=[p for p in pairs if p['pair_type'] in {'hand-box','other-body-box'}]
   worst=max(relevant,key=lambda p:p['penetration_depth'],default=None)
   trace.append({'iteration':i,'max_all_body_object_penetration_mm':1000*worst['penetration_depth'] if worst else 0.,'worst_pair':[worst['geom_a'],worst['geom_b']] if worst else [],'max_non_hand_body_object_penetration_mm':1000*max((p['penetration_depth'] for p in pairs if p['pair_type']=='other-body-box'),default=0.)})
  probe.update(n_iter=budget,min_iterations=budget,iteration_observer=observe)
  if unweighted and probe['vertex_residual_weights'] is not None:probe['vertex_residual_weights']=np.ones_like(probe['vertex_residual_weights'])
  error=None
  try:original(self,**probe)
  except RuntimeError as e:error=str(e)
  row['probes'][name]={'trace':trace,'error':error}
 records.append(row);(args.output/'iteration_probe.json').write_text(json.dumps(records,indent=2)+'\n')
 if frame==max(args.frames):raise FinishedProbe()
 return baseline

InteractionMeshRetargeter.iterate=iterate
sys.argv=[str(root/'tools/run_current_semantic_retarget_comparison.py'),'--execute-retarget',str(replay)]
try:runpy.run_path(sys.argv[0],run_name='__main__')
except FinishedProbe:print('Diagnostic probes saved; no probe trajectory accepted.')
