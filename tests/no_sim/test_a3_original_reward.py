"""A3 Original RL must use the unmodified baseline reward."""

import pytest

from holosoma.config_values.wbt.a3.experiment import (
    a3_31dof_wbt_w_object_b4_s0_original_adaptive_paper_dr,
    a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
)
from holosoma.config_values.wbt.g1.paper_dr import (
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
)

pytestmark = pytest.mark.no_sim


def test_a3_experiments_inherit_original_rl_reward():
    baseline = g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr.reward
    for experiment in (
        a3_31dof_wbt_w_object_b4_s0_original_adaptive_paper_dr,
        a3_31dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
    ):
        assert experiment.reward == baseline
        terms = experiment.reward.terms
        assert terms["action_rate_l2"].func.endswith(":penalty_action_rate")
        assert terms["action_rate_l2"].weight == -0.1
        assert terms["limits_dof_pos"].params["soft_dof_pos_limit"] == 0.9
        assert "reference_contact_position" not in terms
