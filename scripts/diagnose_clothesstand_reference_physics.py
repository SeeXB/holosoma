"""Isolated, non-RL reference/contact diagnostic in the training IsaacSim scene.

Does NOT modify production assets or configs. Six environments compare original
and B4 references under (1) prescribed robot/free object, (2) free robot with the
training PD controller targeting reference joints, and (3) object-only gravity.
Prescribed robot states are rewritten every physics tick, not a feasible policy.
The object is initialized ONCE and is never teleported during the test.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def interpolate(data, key, t, fps, quaternion=False):
    a = np.asarray(data[key])
    x = np.clip(t * fps, 0, len(a) - 1)
    i = np.minimum(x.astype(int), len(a) - 2)
    u = (x - i).reshape((-1,) + (1,) * (a.ndim - 1))
    left, right = a[i].copy(), a[i + 1].copy()
    if quaternion:
        right *= np.where(np.sum(left * right, axis=-1, keepdims=True) < 0, -1, 1)
    out = (1 - u) * left + u * right
    if quaternion:
        out /= np.linalg.norm(out, axis=-1, keepdims=True)
    return out


def audit(sim):
    from pxr import Usd, UsdGeom, UsdPhysics

    obj = sim.scene.rigid_objects["object"]
    out = {"object_prim_path": obj.cfg.prim_path, "colliders": []}
    for name in ("get_masses", "get_inertias", "get_coms", "get_material_properties"):
        try:
            out[name] = getattr(obj.root_physx_view, name)().cpu().tolist()
        except Exception as error:
            out[name] = {"error": str(error)}
    for prim in Usd.PrimRange(sim.sim.stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if "/env_0/" not in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        entry = {"path": path, "type": prim.GetTypeName(),
                 "enabled": UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()}
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            entry["approximation"] = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            entry["vertices"] = len(points)
            entry["local_bounds"] = [points.min(0).tolist(), points.max(0).tolist()]
        entry["physics_attributes"] = {
            a.GetName(): str(a.Get()) for a in prim.GetAttributes()
            if a.GetName().startswith(("physics:", "physx"))
        }
        out["colliders"].append(entry)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--collider", choices=("default", "convex_decomposition"), default="default")
    parser.add_argument("--mass", type=float, default=None, help="Optional diagnostic-only object mass override (kg)")
    parser.add_argument("--original-reference", type=Path, default=None)
    parser.add_argument("--b4-reference", type=Path, default=None)
    parser.add_argument("--object-urdf", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "logs/WholeBodyTracking/20260903_093625-sub9_clothesstand_058_originaltraj_originalrl_s42-locomotion/model_12000.pt",
        help="Checkpoint used only as a source of the unchanged experiment/scene configuration.",
    )
    parser.add_argument("--first-reference-label", choices=("original", "uniform2", "repeat"), default="original")
    parser.add_argument("--second-reference-label", choices=("b4", "uniform2", "repeat"), default="b4")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = copy.deepcopy(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["experiment_config"])
    if (args.original_reference is None) != (args.b4_reference is None):
        raise ValueError("Supply both reference overrides or neither")
    if args.original_reference is not None:
        raw["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"] = str(args.original_reference.resolve())
    if args.object_urdf is not None:
        raw["scene"]["rigid_objects"]["object"]["urdf_file"] = str(args.object_urdf.resolve())
    raw["eval_overrides"].update(headless=True, num_envs=6, disable_logger=True)
    # Retain required manager state but exclude startup mass/CoM/inertia DR and pushes.
    setup = raw["randomization"]["setup_terms"]
    raw["randomization"]["setup_terms"] = {
        k: v for k, v in setup.items() if not k.startswith("randomize_")
    }
    for term in raw["randomization"]["setup_terms"].values():
        if "enabled" in term["params"]:
            term["params"]["enabled"] = False
    raw["randomization"]["step_terms"] = {}
    raw["scene"]["env_spacing"] = 4.0
    raw["logger"]["base_dir"] = str(args.output / "logs")
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.utils.sim_utils import setup_simulation_environment
    cfg = ExperimentConfig(**raw).get_eval_config()
    cfg.save_config(str(args.output / "diagnostic_config.yaml"))
    # Converter-only counterfactual: never edits the URDF or the production spawner.
    if args.collider != "default":
        from holosoma.utils import sim_utils
        original_launcher = sim_utils.setup_isaaclab_launcher

        def launcher(*a, **kw):
            app = original_launcher(*a, **kw)
            from holosoma.simulator.isaacsim import object_spawner
            original_select = object_spawner.select_spawn_cfg

            def select(*a, **kw):
                spawn = original_select(*a, **kw)
                if hasattr(spawn, "collider_type"):
                    spawn.collider_type = args.collider
                    spawn.usd_dir = str(args.output.resolve() / "object_usd")
                    spawn.force_usd_conversion = True
                return spawn

            object_spawner.select_spawn_cfg = select
            return app

        sim_utils.setup_isaaclab_launcher = launcher
    env, device, app = setup_simulation_environment(cfg)
    try:
        sim = env.simulator
        obj = sim.scene.rigid_objects["object"]
        if args.mass is not None:
            masses = obj.root_physx_view.get_masses()
            masses[:] = args.mass
            obj.root_physx_view.set_masses(masses, torch.arange(6, dtype=torch.int32))
        (args.output / "asset_audit.json").write_text(json.dumps(audit(sim), indent=2))
        paths = [ROOT / f"src/holosoma/holosoma/data/motions/tasks/sub9_clothesstand_058/sub9_clothesstand_058_{s}_mj_w_obj.npz"
                 for s in ("original", "semantic_b4")]
        if args.original_reference is not None:
            paths = [args.original_reference.resolve(), args.b4_reference.resolve()]
        refs = []
        for path in paths:
            with np.load(path, allow_pickle=False) as z:
                refs.append({k: z[k].copy() for k in z.files})
        fps = float(refs[0]["fps"].item())
        frame_count = len(refs[0]["joint_pos"])
        assert all(len(r["joint_pos"]) == frame_count and float(r["fps"].item()) == fps for r in refs)
        decimation = round(1 / fps / sim.sim_dt)
        assert np.isclose(decimation * sim.sim_dt, 1 / fps)
        times = np.arange((frame_count - 1) * decimation + 1) * sim.sim_dt
        dense = []
        for r in refs:
            order = [list(r["joint_names"]).index(name) for name in sim.dof_names]
            q = interpolate(r, "joint_pos", times, fps)
            # NPZ includes a leading MuJoCo `world` body. Match by name exactly
            # as MotionLoader does; index zero in the *raw* file is not pelvis.
            root_index = list(r["body_names"]).index(sim.body_names[0])
            assert sim.body_names[0] == "pelvis"
            np.testing.assert_allclose(r["body_pos_w"][:, root_index], r["joint_pos"][:, :3], atol=1e-5)
            root_pos = interpolate(r, "body_pos_w", times, fps)[:, root_index]
            root_quat = interpolate(r, "body_quat_w", times, fps, True)[:, root_index]
            vel = interpolate(r, "body_lin_vel_w", times, fps)[:, root_index]
            ang = interpolate(r, "body_ang_vel_w", times, fps)[:, root_index]
            dense.append({"root": np.concatenate((root_pos, root_quat, vel, ang), axis=-1),
                          "q": q[:, 7:][:, order],
                          "qd": interpolate(r, "joint_vel", times, fps)[:, 6:][:, order],
                          "obj": np.concatenate((interpolate(r, "object_pos_w", times, fps),
                                                  interpolate(r, "object_quat_w", times, fps, True),
                                                  interpolate(r, "object_lin_vel_w", times, fps),
                                                  interpolate(r, "object_ang_vel_w", times, fps)), axis=-1)})
        tensors = {k: torch.tensor(np.stack([dense[e % 2][k] for e in range(6)], axis=1),
                                  dtype=torch.float32, device=device) for k in dense[0]}
        origins = sim.scene.env_origins
        tensors["root"][:, :, :3] += origins
        tensors["obj"][:, :, :3] += origins
        ids = torch.arange(6, device=device)
        imposed_ids = torch.tensor([0, 1, 4, 5], device=device)
        labels = [args.first_reference_label, args.second_reference_label]
        names = [f"{mode}_{label}" for mode in ("prescribed", "pd", "object_only") for label in labels]
        buffers = {k: [] for k in ("pre_root_pos", "pre_root_quat_xyzw", "pre_dof_pos", "object_pos_w",
                                    "object_quat_xyzw", "object_lin_vel_w", "object_ang_vel_w",
                                    "body_pos_w", "contact_force_w", "reference_object_pos_w")}

        def write_robot(i, selected):
            state = tensors["root"][i, selected].clone()
            # Object-only control: no robot within reach of the object.
            state[selected >= 4, 0] += 10.0
            sim._robot.write_root_state_to_sim(state, env_ids=selected)
            sim._robot.write_joint_state_to_sim(tensors["q"][i, selected], tensors["qd"][i, selected],
                                                joint_ids=sim.dof_ids, env_ids=selected)

        def record(i):
            sim.refresh_sim_tensors()
            rs, os_ = sim._robot.data.root_state_w, obj.data.root_state_w
            values = {"pre_root_pos": rs[:, :3], "pre_root_quat_xyzw": rs[:, [4, 5, 6, 3]],
                      "pre_dof_pos": sim.dof_pos, "object_pos_w": os_[:, :3],
                      "object_quat_xyzw": os_[:, [4, 5, 6, 3]], "object_lin_vel_w": os_[:, 7:10],
                      "object_ang_vel_w": os_[:, 10:13], "body_pos_w": sim._rigid_body_pos,
                      "contact_force_w": sim.contact_forces, "reference_object_pos_w": tensors["obj"][i, :, :3]}
            for k, v in values.items():
                buffers[k].append(v.detach().cpu().numpy().copy())

        write_robot(0, ids)
        obj.write_root_state_to_sim(tensors["obj"][0], env_ids=ids)
        sim.scene.write_data_to_sim()
        sim.sim.forward()
        sim.scene.update(sim.sim_dt)
        record(0)
        # Abort instead of drawing conclusions from a frame/order/origin error.
        fk_checks = {}
        for e in range(4):
            r = refs[e % 2]
            expected = r["joint_pos"][0, :3] + origins[e].cpu().numpy()
            np.testing.assert_allclose(buffers["pre_root_pos"][0][e], expected, atol=1e-5)
            common = [name for name in sim.body_names if name in list(r["body_names"])]
            errors = {name: float(np.linalg.norm(
                buffers["body_pos_w"][0][e, sim.body_names.index(name)] - origins[e].cpu().numpy()
                - r["body_pos_w"][0, list(r["body_names"]).index(name)])) for name in common}
            fk_checks[names[e]] = errors
            # URDF and retarget MJCF differ at waist/torso by about 11 mm.
            # Record those asset differences; reject larger mapping errors.
            if max(errors.values()) > 0.015:
                raise RuntimeError(f"Initial simulator/reference FK mismatch: {errors}")
        (args.output / "initial_fk_checks.json").write_text(json.dumps(fk_checks, indent=2))
        # No env.step(): no RL, DR, command advance, termination, or automatic reset.
        for i in range(len(times) - 1):
            write_robot(i, imposed_ids)
            sim.refresh_sim_tensors()
            # Same P controller / torque limits as training, directly targeting q_ref.
            # Joint targets held for 20 ms, prescribed state interpolated at 5 ms.
            target_i = (i // decimation) * decimation
            torques = env.p_gains * (tensors["q"][target_i] - sim.dof_pos) - env.d_gains * sim.dof_vel
            if env.robot_config.control.clip_torques:
                torques = torch.clamp(torques, -env.torque_limits, env.torque_limits)
            torques[imposed_ids] = 0
            sim.apply_torques_at_dof(torques)
            sim.simulate_at_each_physics_step()
            if (i + 1) % decimation == 0:
                record(i + 1)
            if (i + 1) % 200 == 0:
                print(f"DIAGNOSTIC physics step {i + 1}/{len(times) - 1}", flush=True)
        arrays = {k: np.stack(v) for k, v in buffers.items()}
        summary = []
        for e, name in enumerate(names):
            r = refs[e % 2]
            data = {k: v[:, e] for k, v in arrays.items()}
            pos = np.linalg.norm(data["object_pos_w"] - data["reference_object_pos_w"], axis=-1)
            dot = np.abs(np.sum(data["object_quat_xyzw"][:, [3, 0, 1, 2]] * r["object_quat_w"], axis=-1))
            ori = 2 * np.arccos(np.clip(dot, 0, 1))
            data.update(motion_step=np.arange(frame_count), done=np.zeros(frame_count, dtype=bool), object_pos_error_m=pos,
                        object_ori_error_rad=ori)
            for reason in ("bad_ref_pos", "bad_ref_ori", "bad_motion_body_pos", "bad_object_pos", "bad_object_ori"):
                data[reason] = np.zeros(frame_count, dtype=bool)
            md = {"fps": fps, "dof_names": sim.dof_names, "body_names": sim.body_names, "mode": name,
                  "reference": str(paths[e % 2]), "collider_override": args.collider,
                  "object_urdf_override": str(args.object_urdf) if args.object_urdf else None,
                  "mass_override": args.mass, "startup_dr": False, "pushes": False,
                  "object_state_writes": 1, "robot_prescribed_dt": sim.sim_dt if e in (0, 1, 4, 5) else None,
                  "note": "No policy, no termination/reset; prescribed robot is an idealized contact diagnostic."}
            data["_metadata_json"] = json.dumps(md)
            out = args.output / name
            out.mkdir(exist_ok=True)
            np.savez_compressed(out / "rollout.npz", **data)
            bad = np.flatnonzero((pos > 1) | (ori > np.pi / 4))
            summary.append({"mode": name, "first_object_threshold_s": float(bad[0] / fps) if len(bad) else None,
                            "max_position_error_m": float(pos.max()), "final_position_error_m": float(pos[-1]),
                            "max_orientation_error_deg": float(np.rad2deg(ori.max())),
                            "final_orientation_error_deg": float(np.rad2deg(ori[-1])),
                            "object_final_z_m": float(data["object_pos_w"][-1, 2]),
                            "object_max_linear_speed_mps": float(np.linalg.norm(data["object_lin_vel_w"], axis=-1).max())})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("DIAGNOSTIC_COMPLETE " + json.dumps(summary), flush=True)
    except BaseException:
        import traceback
        error = traceback.format_exc()
        (args.output / "ERROR.txt").write_text(error)
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        if app is not None:
            from holosoma.utils.sim_utils import close_simulation_app
            close_simulation_app(app)


if __name__ == "__main__":
    main()
