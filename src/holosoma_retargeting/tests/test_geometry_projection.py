from types import SimpleNamespace
import numpy as np
import pytest
from holosoma_retargeting.config_types.semantic import SemanticRetargetingConfig
from holosoma_retargeting.semantic_keyframes.geometry_projection import project_geometry, GeometryProjectionError


class PlaneFixture:
    """One translating hand against a fixed plane; signed clearance is q[0]."""
    def __init__(self, upper=1.):
        self.semantic_config=SemanticRetargetingConfig()
        self.q_a_indices=np.array([0]);self.laplacian_match_links={'hand':'hand'}
        self.foot_links={};self.foot_lock=SimpleNamespace(enable=False)
        self.activate_foot_sticking=True;self.activate_obj_non_penetration=True
        self.activate_joint_limits=True;self.q_a_lb=np.array([-1.]);self.q_a_ub=np.array([upper])
        self.penetration_tolerance=.001;self.foot_sticking_tolerance=.001;self.step_size=.2
        self._self_collision_tolerance=0.;self._convex_solver_calls=0
        self.object_name='object';self._geom_names=['hand','object']
    def _target_interaction_vertices(self,q,obj):return np.vstack([[q[0],0.,0.],obj])
    def _calc_manipulator_jacobians(self,q,links,obj_frame):
        if not links:return {},{},None
        return {'hand':np.array([[1.],[0.],[0.]])},{'hand':np.array([q[0],0.,0.])},None
    def _update_jacobians_and_phis_from_q(self,q):
        j=np.zeros(len(q));j[0]=1.
        return {(0,1):j},{(0,1):float(q[0])}
    def _compute_self_collision_constraints(self,frame):return {},{}


def pose(x):return np.array([x,0.,0.,1.,0.,0.,0.,.123])

def run(rt,q):return project_geometry(rt,q,q,{},0,np.zeros((1,3)),[[1],[0]],np.ones(2))

def test_feasible_pose_is_bitwise_unchanged_without_convex_solve():
    rt=PlaneFixture();q=pose(.01);result,info=run(rt,q)
    np.testing.assert_array_equal(result,q)
    assert rt._convex_solver_calls==0 and info['iterations']==0 and info['accepted']


def test_hand_penetration_is_repaired_without_moving_locked_coordinates():
    rt=PlaneFixture();q=pose(-.02);result,info=run(rt,q)
    assert result[0]>=-.00101 and info['accepted']
    np.testing.assert_array_equal(result[1:],q[1:])
    assert q[0]==-.02 and info['iterations']==rt._convex_solver_calls


def test_infeasible_joint_and_contact_bounds_raise_instead_of_accepting_penetration():
    rt=PlaneFixture(upper=-.01)
    with pytest.raises(GeometryProjectionError) as e:run(rt,pose(-.02))
    assert not e.value.diagnostics['accepted']


def test_projection_cannot_silently_enable_on_original():
    with pytest.raises(ValueError,match='B4'):
        SemanticRetargetingConfig(mode='original',geometry_projection=True).validate()
