from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from holosoma.config_values.wbt.g1.reward import g1_29dof_wbt_reward_w_object
from holosoma.managers.reward.terms.wbt import SemanticPlanUndesiredContacts, UndesiredContacts
from holosoma.utils.semantic_contacts import (
    G1_BODY_PART_LINKS,
    load_semantic_contact_parts,
    resolve_semantic_contact_links,
)

pytestmark = pytest.mark.no_sim


def test_all_anatomy_parts_resolve_on_authored_g1():
    from holosoma import __file__ as package_file

    root = ET.parse(Path(package_file).parent / "data/robots/g1/main_mesh_collision_halfspherehand.urdf").getroot()
    links = [link.attrib["name"] for link in root.findall("link")]
    assert len(links) == 42
    mapping = resolve_semantic_contact_links(G1_BODY_PART_LINKS, links)
    assert len(mapping) == 18
    assert len(mapping["left_hip"]) == 3
    assert mapping["left_hand"] == ("left_sphere_hand_link",)
    assert "torso_link" in mapping["waist"]


def test_hand_fixed_link_merge_fallback_and_missing_errors():
    assert resolve_semantic_contact_links(["left_hand"], ["left_wrist_yaw_link"]) == {
        "left_hand": ("left_wrist_yaw_link",)
    }
    with pytest.raises(ValueError, match="no link"):
        resolve_semantic_contact_links(["left_elbow"], ["pelvis"])
    with pytest.raises(ValueError, match="unknown"):
        resolve_semantic_contact_links(["left_elbw"], ["left_elbow_link"])


def test_plan_rejects_unknown_and_empty_parts(tmp_path):
    path = tmp_path / "plan.json"
    for names in ([], ["left_elbw"]):
        path.write_text(json.dumps({"events": [{"body_parts": names}]}))
        with pytest.raises(ValueError):
            load_semantic_contact_parts(path)
    path.write_text(json.dumps({"actions": [{"body_parts": ["left_forearm", "torso", "left_elbow"]}]}))
    assert load_semantic_contact_parts(path) == ("left_elbow", "waist")


def test_only_named_contacts_removed_and_original_reward_unchanged(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"events": [{"body_parts": ["left_elbow"]}]}))
    names = ["pelvis", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link"]
    # Two envs: left elbow alone; pelvis+right elbow. History max matters.
    forces = torch.zeros(2, 3, 4, 3)
    forces[0, 0, 1, 0] = 2
    forces[1, 1, 0, 1] = 3
    forces[1, 2, 2, 2] = 2
    env = SimpleNamespace(device="cpu", simulator=SimpleNamespace(body_names=names, contact_forces_history=forces))
    cfg = g1_29dof_wbt_reward_w_object.terms["undesired_contacts"]
    old = UndesiredContacts(cfg, env)
    new = SemanticPlanUndesiredContacts(cfg, env)  # command manager not yet constructed
    env.command_manager = SimpleNamespace(
        get_state=lambda _: SimpleNamespace(
            motion_cfg=SimpleNamespace(sampling_mode="semantic_adaptive", semantic_file=str(plan))
        )
    )
    assert old(env).tolist() == [1, 2]
    assert new(env).tolist() == [0, 2]
    assert new.exempted_contact_body_names == ["left_elbow_link"]
    assert old(env).tolist() == [1, 2]
    # No filesystem accesses in subsequent reward steps.
    plan.unlink()
    assert new(env).tolist() == [0, 2]


def test_empty_penalty_set_and_mode_guard(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"events": [{"body_parts": ["pelvis"]}]}))
    motion = SimpleNamespace(sampling_mode="original_adaptive", semantic_file=str(plan))
    env = SimpleNamespace(
        device="cpu",
        simulator=SimpleNamespace(body_names=["pelvis"], contact_forces_history=torch.ones(2, 3, 1, 3)),
        command_manager=SimpleNamespace(get_state=lambda _: SimpleNamespace(motion_cfg=motion)),
    )
    cfg = g1_29dof_wbt_reward_w_object.terms["undesired_contacts"]
    term = SemanticPlanUndesiredContacts(cfg, env)
    with pytest.raises(ValueError, match="semantic_adaptive"):
        term(env)
    motion.sampling_mode = "semantic_adaptive"
    assert term(env).tolist() == [0, 0]


def test_versioned_preset_changes_only_contact_term_and_requires_plan():
    from holosoma.config_values.wbt.g1.paper_dr import (
        g1_29dof_wbt_w_object_b4_s0_original_adaptive_paper_dr as original,
    )
    from holosoma.config_values.wbt.g1.paper_dr import (
        g1_29dof_wbt_w_object_b4_s2_semantic_adaptive_paper_dr as legacy,
    )
    from holosoma.config_values.wbt.g1.paper_dr import (
        g1_29dof_wbt_w_object_b4_s2_semantic_contacts_paper_dr as new,
    )

    assert original.reward == legacy.reward == g1_29dof_wbt_reward_w_object
    for name, cfg in legacy.reward.terms.items():
        if name == "undesired_contacts":
            assert replace(new.reward.terms[name], func=cfg.func) == cfg
        else:
            assert new.reward.terms[name] == cfg
    assert new.command.setup_terms["motion_command"].params["motion_config"].semantic_file == ""
    assert new.randomization == legacy.randomization
    assert new.termination == legacy.termination
    assert new.observation == legacy.observation
