import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_w_object,
    g1_29dof_wbt_w_object_semantic_e0_base,
    g1_29dof_wbt_w_object_semantic_e1_part,
    g1_29dof_wbt_w_object_semantic_e2_part_rel,
    g1_29dof_wbt_w_object_semantic_keyframe,
    g1_29dof_wbt_w_object_semantic_r0_u2_omni,
    g1_29dof_wbt_w_object_semantic_r1_b4_omni,
    g1_29dof_wbt_w_object_semantic_r2_b4_part,
    g1_29dof_wbt_w_object_semantic_r3_b4_part_rel,
    g1_29dof_wbt_w_object_semantic_r4_b4_full,
)
from holosoma.config_values.wbt.g1.paper_dr import (
    g1_29dof_wbt_w_object_b4_omni_paper_dr,
    g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr,
    g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr,
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
    g1_29dof_wbt_w_object_b4_semantic_paper_dr,
)
from holosoma.utils.config_registry import ConfigRegistry

EXPERIMENT_REGISTRY = ConfigRegistry(ExperimentConfig, group="holosoma.config.experiment")

EXPERIMENT_REGISTRY.add("g1_29dof", g1_29dof)
EXPERIMENT_REGISTRY.add("g1_29dof_fast_sac", g1_29dof_fast_sac)
EXPERIMENT_REGISTRY.add("t1_29dof", t1_29dof)
EXPERIMENT_REGISTRY.add("t1_29dof_fast_sac", t1_29dof_fast_sac)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt", g1_29dof_wbt)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object", g1_29dof_wbt_w_object)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_fast_sac", g1_29dof_wbt_fast_sac)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_fast_sac_w_object", g1_29dof_wbt_fast_sac_w_object)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_e0_base", g1_29dof_wbt_w_object_semantic_e0_base)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_e1_part", g1_29dof_wbt_w_object_semantic_e1_part)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_semantic_e2_part_rel",
    g1_29dof_wbt_w_object_semantic_e2_part_rel,
)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_keyframe", g1_29dof_wbt_w_object_semantic_keyframe)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_r0_u2_omni", g1_29dof_wbt_w_object_semantic_r0_u2_omni)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_r1_b4_omni", g1_29dof_wbt_w_object_semantic_r1_b4_omni)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_r2_b4_part", g1_29dof_wbt_w_object_semantic_r2_b4_part)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_semantic_r3_b4_part_rel",
    g1_29dof_wbt_w_object_semantic_r3_b4_part_rel,
)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_semantic_r4_b4_full", g1_29dof_wbt_w_object_semantic_r4_b4_full)
EXPERIMENT_REGISTRY.add("g1_29dof_wbt_w_object_b4_omni_paper_dr", g1_29dof_wbt_w_object_b4_omni_paper_dr)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_b4_semantic_paper_dr",
    g1_29dof_wbt_w_object_b4_semantic_paper_dr,
)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr",
    g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr,
)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr",
    g1_29dof_wbt_w_object_b4_s1_semantic_uniform_paper_dr,
)
EXPERIMENT_REGISTRY.add(
    "g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr",
    g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr,
)


def get_annotated_experiment_config() -> type:
    """Return the ``exp:`` subcommand type."""
    return Annotated[  # type: ignore[return-value]  # Annotated[...] alias; mypy models it as object
        ExperimentConfig,
        tyro.conf.arg(
            constructor=tyro.extras.subcommand_type_from_defaults(
                {f"exp:{k.replace('_', '-')}": v for k, v in EXPERIMENT_REGISTRY.items()}
            )
        ),
    ]


from holosoma.utils.config_registry import (  # noqa: E402
    deprecated_defaults_alias as _deprecated_defaults_alias,
)

__getattr__ = _deprecated_defaults_alias(__name__, EXPERIMENT_REGISTRY)
