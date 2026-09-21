"""Run retarget CLI and audit first infeasible subproblem without accepting a relaxed solution.

Usage: python script.py --audit-output /path/audit.json [normal retarget CLI args...]
"""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import sys

import cvxpy as cp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--audit-output", type=Path, required=True)
parser.add_argument("--run-config-job", type=Path, help="Replay a B4 comparison job through its Python configuration entrypoint.")
args, rest = parser.parse_known_args()
original_solve = cp.Problem.solve
reported = False


def solve(problem, *a, **kw):
    global reported
    result = original_solve(problem, *a, **kw)
    caller = inspect.currentframe().f_back
    local = caller.f_locals
    # Frame-zero trust-region failure has an existing built-in fallback. Audit
    # unrecovered later failures, not that transient initialization attempt.
    if (not reported and problem.status.startswith("infeasible")
            and "original_constraint_objects" in local and not local.get("init_t", False)):
        reported = True
        groups = local["original_constraint_objects"]
        rt = local["self"]
        report = {"frame": local["frame_idx"], "status": problem.status, "step_size": rt.step_size,
                  "constraint_counts": {k: len(v) for k, v in groups.items()}, "omit_group": {},
                  "contacts": []}
        for name, constraints in groups.items():
            excluded = {id(c) for c in constraints}
            test = cp.Problem(problem.objective, [c for c in problem.constraints if id(c) not in excluded])
            original_solve(test, *a, **kw)
            report["omit_group"][name] = test.status
        for pair, phi in local.get("phis", {}).items():
            report["contacts"].append({"pair": [rt._geom_names[i] for i in pair], "distance_m": float(phi),
                                       "jacobian_norm": float(np.linalg.norm(local["Js"][pair][rt.q_a_indices]))})
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(report, indent=2))
        np.savez_compressed(args.audit_output.with_suffix(".npz"), qpos=local["q"],
                            previous_qpos=local["q_t_last"])
        print("INFEASIBILITY_AUDIT " + json.dumps(report), flush=True)
    return result


cp.Problem.solve = solve
if args.run_config_job:
    runner = Path(__file__).resolve().parents[1] / "tools/run_current_semantic_retarget_comparison.py"
    sys.argv = [str(runner), "--execute-retarget", str(args.run_config_job)]
    runpy.run_path(str(runner), run_name="__main__")
else:
    sys.argv = [sys.argv[0]] + rest
    runpy.run_module("holosoma_retargeting.examples.robot_retarget", run_name="__main__")
