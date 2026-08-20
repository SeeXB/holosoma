"""Default reward manager configurations."""

from holosoma.config_types.reward import RewardManagerCfg
from holosoma.config_values.loco.g1.reward import g1_29dof_loco, g1_29dof_loco_fast_sac
from holosoma.config_values.loco.t1.reward import t1_29dof_loco, t1_29dof_loco_fast_sac
from holosoma.config_values.wbt.g1.reward import (
    g1_29dof_wbt_fast_sac_reward,
    g1_29dof_wbt_reward,
    g1_29dof_wbt_reward_w_object,
    g1_29dof_wbt_semantic_keyframe_reward,
    g1_29dof_wbt_semantic_reward,
    g1_29dof_wbt_w_object_semantic_e1_part_reward,
    g1_29dof_wbt_w_object_semantic_e2_part_rel_reward,
    g1_29dof_wbt_w_object_semantic_keyframe_reward,
    g1_29dof_wbt_w_object_semantic_reward,
)
from holosoma.utils.config_registry import ConfigRegistry

REWARD_REGISTRY = ConfigRegistry(RewardManagerCfg, group="holosoma.config.reward")

none = REWARD_REGISTRY.add("none", None)
REWARD_REGISTRY.add("t1_29dof_loco", t1_29dof_loco)
REWARD_REGISTRY.add("t1_29dof_loco_fast_sac", t1_29dof_loco_fast_sac)
REWARD_REGISTRY.add("g1_29dof_loco", g1_29dof_loco)
REWARD_REGISTRY.add("g1_29dof_loco_fast_sac", g1_29dof_loco_fast_sac)
REWARD_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt_reward)
REWARD_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_reward_w_object)
REWARD_REGISTRY.add("g1_29dof_wbt_fast_sac", g1_29dof_wbt_fast_sac_reward)
REWARD_REGISTRY.add("g1_29dof_wbt_semantic_keyframe", g1_29dof_wbt_semantic_keyframe_reward)
REWARD_REGISTRY.add("g1_29dof_wbt_semantic_reward", g1_29dof_wbt_semantic_reward)
REWARD_REGISTRY.add("g1_29dof_wbt_w_object_semantic_reward", g1_29dof_wbt_w_object_semantic_reward)
REWARD_REGISTRY.add(
    "g1_29dof_wbt_w_object_semantic_e1_part",
    g1_29dof_wbt_w_object_semantic_e1_part_reward,
)
REWARD_REGISTRY.add(
    "g1_29dof_wbt_w_object_semantic_e2_part_rel",
    g1_29dof_wbt_w_object_semantic_e2_part_rel_reward,
)
REWARD_REGISTRY.add(
    "g1_29dof_wbt_w_object_semantic_keyframe",
    g1_29dof_wbt_w_object_semantic_keyframe_reward,
)

from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, REWARD_REGISTRY)
