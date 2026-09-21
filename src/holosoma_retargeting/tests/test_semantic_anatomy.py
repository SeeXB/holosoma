from __future__ import annotations

import json
import numpy as np
from holosoma_retargeting.semantic_keyframes.pipeline import (
    MAPPABLE_BODY_PARTS,
    build_dynamic_prompt,
    dynamic_plan_validation_issues,
    load_retargeting_bundle_signals,
    normalize_dynamic_plan,
)
from holosoma_retargeting.semantic_keyframes.runtime import resolve_body_vertex_mapping
from holosoma_retargeting.semantic_keyframes.trajectory_events import load_event_signals


def test_anatomy_normalization_retains_unknown_for_validation():
    result = normalize_dynamic_plan(
        {"actions": [{"body_parts": ["left_elbow", "torso", "right_forearm", "left_elbw"]}]}
    )
    assert result["actions"][0]["body_parts"] == ["left_elbow", "waist", "right_elbow", "left_elbw"]
    result["actions"][0]["action"] = "support_table"
    assert any("left_elbw" in issue["message"] for issue in dynamic_plan_validation_issues(result))
    assert len(MAPPABLE_BODY_PARTS) == 18


def test_body_only_signals_no_fabricated_object(tmp_path):
    joints = np.zeros((11, 3, 3))
    joints[:, 0, 0] = np.linspace(0, 1, 11)
    joints[:, 0, 2] = 1
    joints[:, 1, 2] = 0.8
    joints[:, 2, 2] = 0.4
    path = tmp_path / "body.npz"
    np.savez(
        path,
        human_joints=joints,
        smplh_joint_names=np.asarray(["Hips", "LeftForeArm", "LeftHand"]),
        human_joint_layout=np.asarray("lafan_z_up_v1"),
        frame_ids=np.arange(11),
        fps=10.0,
    )
    signals, thresholds, fps = load_retargeting_bundle_signals(path)
    assert fps == 10 and thresholds == {}
    assert not any("object" in key for key in signals)
    np.testing.assert_allclose(signals["pelvis_speed"], 1.0)
    np.testing.assert_allclose(signals["left_elbow_height"], 0.8)
    np.testing.assert_allclose(signals["body_travel_progress"], np.linspace(0, 1, 11))
    prompt = build_dynamic_prompt(available_signals=signals, body_only=True)
    assert "There is no manipulated object" in prompt
    assert "left_elbow" in prompt


def test_object_rotate_accumulates_quaternion_orientation_travel(tmp_path):
    angles = np.deg2rad([0.0, 90.0, 180.0, 270.0])
    quaternions = np.column_stack([
        np.cos(angles / 2), np.zeros(4), np.zeros(4), np.sin(angles / 2)
    ])
    # Quaternion sign changes must not create a fictitious full revolution.
    quaternions[2] *= -1
    object_poses = np.column_stack([quaternions, np.zeros((4, 3))])
    joints = np.zeros((4, 3, 3))
    path = tmp_path / "object_rotation.npz"
    np.savez(
        path,
        human_joints=joints,
        object_poses_wxyz_xyz=object_poses,
        smplh_joint_names=np.asarray(["L_Middle3", "R_Middle3", "Pelvis"]),
        frame_ids=np.arange(4),
        fps=10.0,
    )

    signals, units, fps = load_event_signals(path)

    assert fps == 10.0
    np.testing.assert_allclose(
        signals["object_rotate"], np.deg2rad([0.0, 90.0, 180.0, 270.0]), atol=1e-7
    )
    assert units["object_rotate"].startswith("rad;")


def test_anatomy_runtime_mapping_uses_limb_joints():
    mapping = resolve_body_vertex_mapping(
        ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Shoulder", "L_Elbow", "L_Wrist"],
        [
            "pelvis",
            "waist",
            "left_hip",
            "left_knee",
            "left_ankle",
            "left_shoulder",
            "left_elbow",
            "left_wrist",
            "left_hand",
        ],
    )
    assert not mapping.missing
    assert mapping.joint_names["left_elbow"] == "L_Elbow"
    assert mapping.joint_names["waist"] == "Pelvis"


def test_dynamic_prompt_contains_complete_json_event_example_with_rotation():
    prompt = build_dynamic_prompt(
        object_name="chair",
        available_signals={"object_height", "object_rotate", "object_angular_speed"},
    )
    example_start = prompt.index('{\n  "actions"')
    example = json.JSONDecoder().raw_decode(prompt[example_start:])[0]
    assert set(example) == {"actions"}
    assert len(example["actions"]) == 3
    rotate = example["actions"][1]
    assert rotate["action"] == "example_rotate_object"
    assert rotate["keyframe_function"]["start"]["signal"] == "object_rotate"
    place = example["actions"][2]
    assert place["keyframe_function"]["start"]["primitive"] == "threshold_crossing_down"
    assert place["keyframe_function"]["end"]["threshold"]["q"] == 0.2
    assert "Do not return Python" in prompt


def test_batch_quota_error_blocks_queued_jobs_without_api(tmp_path, monkeypatch):
    import threading

    import generate_remaining37_semantic_plans as batch
    from holosoma_retargeting.semantic_keyframes.pipeline import VLMQuotaError

    bundle = tmp_path / "data/omomo/bundles/test/input/omomo_gt_sequence.npz"
    bundle.parent.mkdir(parents=True)
    bundle.write_bytes(b"test source")
    monkeypatch.setattr(batch, "DATA", tmp_path / "data")
    monkeypatch.setattr(batch, "load_retargeting_bundle_signals", lambda _: ({"pelvis_speed": np.ones(10)}, {}, 30))
    calls = []

    def exhausted(**kwargs):
        calls.append(kwargs)
        raise VLMQuotaError("insufficient_quota")

    monkeypatch.setattr(batch, "generate_localized_semantic_keyframes", exhausted)
    stop = threading.Event()
    first = batch.generate_one("omomo", "test", tmp_path / "out", tmp_path / "audit", 1, False, stop_event=stop)
    second = batch.generate_one("omomo", "queued", tmp_path / "out", tmp_path / "audit", 1, False, stop_event=stop)
    assert first["status"] == "failed" and stop.is_set()
    assert second["status"] == "blocked" and len(calls) == 1
