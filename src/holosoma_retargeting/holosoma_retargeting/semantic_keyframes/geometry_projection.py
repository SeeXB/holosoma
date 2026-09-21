"""Opt-in local feasibility projection preserving the B4 interaction geometry.

No contact exemptions or relaxed accepted constraints. Trial convex solutions are
checked against nonlinear geometry after quaternion normalization. Failure is
explicit; callers must not export the uncorrected frame as a successful result.
"""
from __future__ import annotations

import time
import cvxpy as cp
import numpy as np
from scipy import sparse


class GeometryProjectionError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def project_geometry(rt, q_base, q_previous, foot_sticking, frame, obj_points, adjacency, weights):
    # Same uniform Laplacian definition as the main optimizer.
    from holosoma_retargeting.src.utils import calculate_laplacian_matrix

    cfg = rt.semantic_config
    timing = {'geometry_check_time_s': 0., 'convex_solve_time_s': 0., 'linearization_time_s': 0., 'checks': 0}
    eps = cfg.geometry_projection_numerical_tolerance
    indices = rt.q_a_indices
    keys = list(rt.laplacian_match_links)
    base_vertices = rt._target_interaction_vertices(q_base, obj_points)
    L = sparse.csr_matrix(calculate_laplacian_matrix(base_vertices, adjacency))
    K = sparse.kron(L, sparse.eye(3), format='csr')
    reference = (L @ base_vertices).reshape(-1)
    alpha = np.ones(len(base_vertices)) if weights is None else np.asarray(weights)
    sqrt_alpha = np.sqrt(np.repeat(alpha, 3))
    _, anchors, _ = rt._calc_manipulator_jacobians(q_previous, links=rt.foot_links, obj_frame=False)
    active_sides = {side: any(bool(v) and k.lower().startswith(side[0]) for k,v in foot_sticking.items()) for side in ('left','right')}

    def foot_bounds(q):
        jac, pos, _ = rt._calc_manipulator_jacobians(q, links=rt.foot_links, obj_frame=False)
        result = []
        for name, point in pos.items():
            active = rt.activate_foot_sticking and any(side in name and enabled for side,enabled in active_sides.items())
            if active:
                for axis in (0,1): result.append((jac[name][axis], float(point[axis]-anchors[name][axis]), rt.foot_sticking_tolerance))
            z = rt._is_foot_locked_in_window(name,frame) if rt.foot_lock.enable else None
            if z is not None: result.append((jac[name][2], float(point[2]-z),rt.foot_lock.tolerance))
        return result

    def measure(q):
        started = time.perf_counter()
        Js, phi = rt._update_jacobians_and_phis_from_q(q)
        collision = [max(0.,-float(v)-rt.penetration_tolerance) for v in phi.values()] if rt.activate_obj_non_penetration else []
        Jself, pself = rt._compute_self_collision_constraints(frame)
        self_errors = [max(0.,rt._self_collision_tolerance-float(v)) for v in pself.values()]
        feet = foot_bounds(q)
        foot_errors = [max(0.,abs(v)-tol) for _,v,tol in feet]
        joint_errors = np.concatenate([np.maximum(rt.q_a_lb-q[indices],0.),np.maximum(q[indices]-rt.q_a_ub,0.)]) if rt.activate_joint_limits else np.zeros(0)
        residual = np.array(collision+self_errors+foot_errors+list(joint_errors))
        feasible = max(collision+self_errors+foot_errors,default=0.) <= eps and np.max(joint_errors,initial=0.) <= 1e-6
        timing['geometry_check_time_s'] += time.perf_counter()-started
        timing['checks'] += 1
        return dict(feasible=bool(feasible), merit=float(np.dot(residual,residual)), max_violation_m=float(np.max(residual,initial=0.)),
                    max_environment_penetration_m=max((-float(v) for v in phi.values()),default=0.),
                    worst_pair=[rt._geom_names[k] for k in min(phi,key=phi.get)] if phi else [],
                    max_foot_excess_m=max(foot_errors,default=0.),max_joint_excess=float(np.max(joint_errors,initial=0.))), (Js,phi,Jself,pself,feet)

    q=np.array(q_base,copy=True)
    initial,_=measure(q)
    info={'frame':frame,'iterations':0,'initial':initial,'trace':[],'accepted':initial['feasible'],'timing':timing}
    if initial['feasible']:
        info['final']=initial
        return q,info
    for iteration in range(cfg.geometry_projection_max_iterations):
        state,(Js,phi,Jself,pself,feet)=measure(q)
        linearization_started=time.perf_counter()
        jac,pos,_=rt._calc_manipulator_jacobians(q,links=rt.laplacian_match_links,obj_frame=(rt.object_name!='ground'))
        vertices=np.vstack([[pos[k] for k in keys],obj_points])
        J=np.zeros((3*len(vertices),len(indices)))
        for i,key in enumerate(keys):J[3*i:3*i+3]=jac[key]
        delta=cp.Variable(len(indices))
        residual=(L @ vertices).reshape(-1)-reference+(K @ J) @ delta
        objective=cp.sum_squares(cp.multiply(sqrt_alpha,residual)) + 1e-3*cp.sum_squares(q[indices]+delta-q_base[indices]) + .1*cp.sum_squares(delta)
        constraints=[cp.SOC(rt.step_size,delta)]
        quat=[(i,int(k)) for i,k in enumerate(indices) if 3<=k<7]
        if quat:constraints += [sum(q[k]*delta[i] for i,k in quat)==0]
        if rt.activate_obj_non_penetration:
            constraints += [Js[k][indices] @ delta >= -v-rt.penetration_tolerance+min(2e-4,rt.penetration_tolerance/5) for k,v in phi.items()]
        constraints += [Jself[k][indices] @ delta >= rt._self_collision_tolerance-v for k,v in pself.items()]
        for j,v,tol in feet:constraints += [j @ delta+v <= tol,j @ delta+v >= -tol]
        if rt.activate_joint_limits:constraints += [q[indices]+delta>=rt.q_a_lb,q[indices]+delta<=rt.q_a_ub]
        problem=cp.Problem(cp.Minimize(1000*objective),constraints)
        timing['linearization_time_s'] += time.perf_counter()-linearization_started
        rt._convex_solver_calls += 1
        info['iterations'] += 1
        solve_started=time.perf_counter()
        try:
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError as exc:
            info['reason']=str(exc);break
        finally:
            timing['convex_solve_time_s'] += time.perf_counter()-solve_started
        if problem.status not in (cp.OPTIMAL,cp.OPTIMAL_INACCURATE):
            info['solver_status']=str(problem.status)
            break
        accepted=None
        for scale in (2.**-i for i in range(13)):
            candidate=q.copy();candidate[indices] += scale*delta.value
            candidate[3:7] /= np.linalg.norm(candidate[3:7])
            candidate_state,_=measure(candidate)
            # All constraints are rechecked. No global penetration slack is accepted.
            if candidate_state['feasible'] or candidate_state['merit'] < state['merit']*(1.-1e-4*scale):
                accepted=(candidate,candidate_state,scale);break
        if accepted is None:
            info['reason']='no_nonlinear_progress';break
        q,state,scale=accepted
        info['trace'].append({'iteration':iteration+1,'step_scale':scale,**state})
        if state['feasible']:
            info.update(accepted=True,final=state)
            return q,info
    info['final']=measure(q)[0]
    raise GeometryProjectionError(f'Geometry projection failed at frame {frame}',info)
