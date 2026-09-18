"""Supervise three independent train -> Paper-DR eval -> offline-video jobs.

Run under the hssim environment in a persistent tmux session. Each worker
evaluates immediately after its own training process exits successfully.
State, commands and exit codes are persisted for recovery and inspection.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback
import uuid
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SUFFIXES = {1: "originaltraj_originalrl", 2: "semanticb4traj_originalrl", 3: "semanticb4traj_semanticadaptive"}


def now():
    return datetime.now().astimezone().isoformat()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def run_process(command, log, state, state_path, phase, env=None, timeout=None):
    state.update(phase=phase, updated=now(), command=[str(x) for x in command])
    save_json(state_path, state)
    with Path(log).open("a") as output:
        output.write(f"\n{now()} {json.dumps(state['command'])}\n")
        output.flush()
        proc = subprocess.Popen(state["command"], cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        state.update(pid=proc.pid)
        save_json(state_path, state)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            code = 124
    state.update(pid=None, updated=now(), **{f"{phase}_exit_code": code})
    save_json(state_path, state)
    if code:
        raise RuntimeError(f"{phase} exited {code}; inspect {log}")


def train_command(cfg, group, train_base, run_id):
    preset = "s2-semantic-adaptive" if group == 3 else "s0-original-adaptive"
    motion = cfg["original_motion"] if group == 1 else cfg["b4_motion"]
    command = [sys.executable, "src/holosoma/holosoma/train_agent.py",
               f"exp:g1-29dof-wbt-w-object-b4-{preset}-paper-dr",
               "logger:wandb" if cfg.get("wandb", True) else "logger:disabled",
               "--training.headless", "True", "--training.name", f"{cfg['task']}_{SUFFIXES[group]}_s{cfg['seed']}_{cfg['tag']}",
               "--training.seed", cfg["seed"], "--training.num-envs", cfg["num_envs"],
               "--algo.config.num-learning-iterations", cfg["iterations"],
               "--training.export-onnx", "True", "--logger.base-dir", train_base,
               "--logger.video.enabled", "False", "--logger.video.upload-to-wandb", "False",
               "--scene.rigid-objects.object.urdf-file", cfg["object_urdf"],
               "--command.setup-terms.motion-command.params.motion-config.motion-file", motion]
    if cfg.get("wandb", True):
        command += ["--logger.id", run_id, "--logger.group", f"{cfg['task']}_{cfg['tag']}"]
    if group == 3:
        command += ["--command.setup-terms.motion-command.params.motion-config.semantic-file", cfg["semantic_file"]]
    return command


def eval_command(cfg, checkpoint, out, motion):
    import torch
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["experiment_config"]
    stored_motion = saved["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]["motion_file"]
    if Path(stored_motion).resolve() != Path(motion).resolve():
        raise ValueError(f"Checkpoint reference mismatch: {stored_motion} != {motion}")
    with np.load(motion, allow_pickle=False) as data:
        frames, fps = len(data["joint_pos"]) - 1, float(data["fps"].item())
    return [sys.executable, "src/holosoma/holosoma/eval_agent.py", "--checkpoint", checkpoint,
            "--import-file", "scripts/eval_paper_dr_robustness_instrumentation.py",
            "--recording.config.enabled", "--recording.config.output-path", out / "rollout.npz",
            "--eval-overrides.headless", "True", "--training.headless", "True", "--training.num-envs", 32,
            "--training.seed", cfg["seed"], "--training.max-eval-steps", 10 * (frames + 2) + 20,
            "--training.export-onnx", "False", "--simulator.config.sim.max-episode-length-s", frames / fps,
            "--command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale", 0.0,
            "--termination.terms.bad-tracking.params.bad-object-pos-threshold", 1.0,
            "--termination.terms.bad-tracking.params.bad-object-ori-threshold", np.pi / 4,
            "--logger.headless-recording", "False", "--logger.video.enabled", "False",
            "--logger.video.upload-to-wandb", "False"]


def upload_report(config_path, result_path):
    import wandb
    cfg = json.loads(Path(config_path).read_text())
    result_path = Path(result_path)
    report = json.loads(result_path.read_text())
    results = report["results"] if "results" in report else [report]
    label = "summary" if len(results) == 3 else f"group{results[0]['group']}"
    with wandb.init(entity="yumou0319-", project="WholeBodyTracking", mode="online",
                    name=f"{cfg['task']}_{cfg['tag']}_{label}_eval", group=f"{cfg['task']}_{cfg['tag']}_eval",
                    dir=str(result_path.parent), config={"task": cfg["task"], "seed": cfg["seed"],
                    "num_envs": 32, "episodes_per_env": 10, "checkpoint_iteration": cfg["iterations"] - 1,
                    "protocol": "paper_dr_robustness_v1", "shape_scale_implemented": False}) as run:
        metrics = {}
        for result in results:
            metrics.update({f"group{result['group']}/{key}": result[key] for key in
                            ("success_rate", "successes", "episodes_scored", "mean_episode_seconds", "mean_reached_frame")})
            metrics.update({f"group{result['group']}/{key}": val for key,val in result["trajectory_errors"].items()})
        run.log(metrics)
        url = run.url
    (result_path.parent / "wandb_url.txt").write_text(url + "\n")


def worker(cfg, config_path, group):
    work = Path(cfg["work_root"]) / f"group{group}"
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {
        "group": group, "started": now(), "train_run_id": uuid.uuid4().hex[:8]}
    train_base = work / "train"
    out = Path(cfg["eval_root"]) / f"group{group}_{cfg['iterations'] - 1:05d}"
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg.get("gpu", 0)))
    motion = cfg["original_motion"] if group == 1 else cfg["b4_motion"]
    try:
        if not state.get("training_complete"):
            if state.get("phase"):
                raise RuntimeError("Existing incomplete training state; inspect before explicitly resuming training")
            run_process(train_command(cfg, group, train_base, state["train_run_id"]), work / "train.log",
                        state, state_path, "training", env)
            matches = list(train_base.rglob(f"model_{cfg['iterations'] - 1:05d}.pt"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one final checkpoint, found {matches}")
            state.update(training_complete=True, checkpoint=str(matches[0].resolve()), training_finished=now())
            save_json(state_path, state)
        checkpoint = Path(state["checkpoint"])
        if not state.get("evaluation_complete"):
            run_process(eval_command(cfg, checkpoint, out, motion), out / "eval.log", state, state_path, "evaluation", env)
            from scripts.analyze_sub10_largebox_089_final import summarize_run
            result = summarize_run(out, 10, cfg["iterations"] - 1, {group: checkpoint.parent}, reference_file=Path(motion))
            if result["episodes_scored"] != 320 or result["num_envs"] != 32:
                raise RuntimeError("Incomplete evaluation quota")
            save_json(out / "result.json", result)
            state.update(evaluation_complete=True, evaluation_finished=now(), success_rate=result["success_rate"])
            save_json(state_path, state)
        # A rendering/upload failure must not discard completed physics metrics.
        if not state.get("video_complete"):
            try:
                render_env = dict(env, PATH="/usr/bin:" + env.get("PATH", ""))
                run_process([cfg["render_python"], "scripts/render_clothesstand_checkpoint_eval.py",
                             "--actual", out / "rollout.npz", "--reference", motion, "--scene", cfg["scene_xml"],
                             "--label", f"Group {group}: {SUFFIXES[group]}", "--output", out / "actual_vs_reference.mp4"],
                            out / "render.log", state, state_path, "rendering", render_env, timeout=1800)
                state["video_complete"] = True
            except Exception as error:
                state["video_error"] = str(error)
        if cfg.get("wandb", True) and not state.get("upload_complete"):
            for attempt in range(3):
                try:
                    run_process([sys.executable, __file__, "--config", config_path, "--upload", out / "result.json"],
                                out / "upload.log", state, state_path, "upload", env, timeout=180)
                    state["upload_complete"] = True
                    state.pop("upload_error", None)
                    break
                except Exception as error:
                    state["upload_error"] = str(error)
        state.update(phase="complete" if state.get("video_complete") and (not cfg.get("wandb", True) or state.get("upload_complete"))
                     else "evaluation_complete_postprocessing_pending", updated=now())
        save_json(state_path, state)
        return state
    except Exception as error:
        state.update(phase="failed", error=str(error), traceback=traceback.format_exc(), updated=now())
        save_json(state_path, state)
        print(f"{now()} group={group} FAILED: {error}", flush=True)
        return state


def preflight(cfg):
    for key in ("original_motion", "b4_motion", "semantic_file", "object_urdf", "scene_xml", "render_python"):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f"{key}: {cfg[key]}")
    references = []
    for key in ("original_motion", "b4_motion"):
        with np.load(cfg[key], allow_pickle=False) as data:
            if not all(np.isfinite(data[k]).all() for k in data.files if data[k].dtype.kind in "fci"):
                raise ValueError(f"Non-finite motion: {key}")
            references.append((len(data["joint_pos"]), float(data["fps"].item())))
    if references[0] != references[1]:
        raise ValueError(f"Reference length/FPS mismatch: {references}")
    from holosoma.managers.command.semantic_transition_sampler import load_semantic_transitions
    load_semantic_transitions(cfg["semantic_file"], motion_fps=references[0][1], motion_time_step_total=references[0][0])
    inputs = {k: Path(cfg[k]) for k in ("original_motion", "b4_motion", "semantic_file", "object_urdf", "scene_xml")}
    # URDF paths alone do not protect against edits to their collision meshes.
    for mesh in ET.parse(cfg["object_urdf"]).iter("mesh"):
        asset = Path(mesh.attrib["filename"])
        if not asset.is_absolute():
            asset = Path(cfg["object_urdf"]).parent / asset
        inputs[str(asset.resolve())] = asset
    return {k: hashlib.sha256(path.read_bytes()).hexdigest() for k, path in inputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--upload", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.upload:
        upload_report(args.config, args.upload)
        return
    hashes = preflight(cfg)
    if args.check_only:
        print(json.dumps({"preflight": "passed", "input_sha256": hashes}, indent=2))
        return
    root = Path(cfg["work_root"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = root / "input_sha256.json"
    if saved.exists() and json.loads(saved.read_text()) != hashes:
        raise ValueError("Experiment inputs changed since launch")
    spec = root / "experiment.json"
    if spec.exists() and json.loads(spec.read_text()) != cfg:
        raise ValueError("Experiment configuration changed since launch")
    save_json(saved, hashes)
    save_json(root / "experiment.json", cfg)
    print(f"{now()} supervisor pid={os.getpid()}; each group evaluates immediately after successful training", flush=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(worker, cfg, args.config.resolve(), g) for g in (1,2,3)]
        states = [f.result() for f in as_completed(futures)]
    if not all(s.get("evaluation_complete") for s in states):
        raise SystemExit("One or more groups failed; inspect group*/state.json")
    eval_root = Path(cfg["eval_root"])
    results = [json.loads((eval_root / f"group{g}_{cfg['iterations'] - 1:05d}/result.json").read_text()) for g in (1,2,3)]
    report = {"task": cfg["task"], "protocol": "Paper-DR, seed 42, 32 environments x first 10 completed episodes; pushes on; initial-pose noise off; object thresholds 1m/45deg; shape scaling unsupported.", "results": results}
    save_json(eval_root / "final_results.json", report)
    lines = [f"# {cfg['task']} final evaluation", "", report["protocol"], "",
             "| Group | Successes | SR | Object RMSE (m) | Orientation RMSE (deg) | Body RMSE (m) |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        e = r["trajectory_errors"]
        lines.append(f"| {r['label']} | {r['successes']}/320 | {100*r['success_rate']:.2f}% | {e['object_pos_rmse_m']:.4f} | {e['object_ori_rmse_deg']:.2f} | {e['tracked_body_pos_rmse_m']:.4f} |")
    (eval_root / "FINAL_EVAL_REPORT.md").write_text("\n".join(lines) + "\n")
    if cfg.get("wandb", True):
        summary_state = {}
        run_process([sys.executable, __file__, "--config", args.config.resolve(), "--upload", eval_root / "final_results.json"],
                    eval_root / "upload.log", summary_state, eval_root / "upload_state.json", "upload", timeout=180)
    print(f"{now()} all three evaluations complete: {eval_root}", flush=True)


if __name__ == "__main__":
    main()
