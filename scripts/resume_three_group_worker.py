"""Recover a stopped worker without restarting its running sibling groups."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_three_group_experiment import (
    ROOT, now, run_process, save_json, train_command, worker,
)


def recovery_command(cfg, group, work, checkpoint, completed_iteration, run_id):
    remaining = cfg["iterations"] - completed_iteration - 1
    if remaining <= 0:
        raise ValueError("Checkpoint already reached the requested training length")
    resume_cfg = dict(cfg, iterations=remaining, save_interval=500)
    command = train_command(resume_cfg, group, work / "train", run_id)
    command += ["--training.checkpoint", str(checkpoint),
                "--algo.config.load-optimizer", "True"]
    return command


def runtime_config(cfg, gpu):
    """Override placement without changing the frozen experiment specification."""
    if gpu is not None and gpu < 0:
        raise ValueError("GPU index must be nonnegative")
    return dict(cfg) if gpu is None else dict(cfg, gpu=gpu)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--completed-iteration", type=int, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--allow-paused", action="store_true", help="Explicitly resume a user-paused worker")
    parser.add_argument("--skip-finalization", action="store_true", help="Another worker owns the joint report")
    parser.add_argument("--reason", default="Recovery after training failure")
    parser.add_argument("--gpu", type=int, help="Runtime GPU override; keep the original experiment immutable")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(cfg["work_root"])
    work = root / f"group{args.group}"
    state_path = work / "state.json"
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_relative_to((work / "train").resolve()) or not checkpoint.is_file():
        raise ValueError("Recovery checkpoint must belong to this worker")
    if cfg != json.loads((root / "experiment.json").read_text()):
        raise ValueError("Experiment configuration changed since original launch")
    original_gpu = cfg["gpu"]
    cfg = runtime_config(cfg, args.gpu)
    lock = (work / "recovery.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(state_path.read_text())
    allowed_phases = {"failed", "paused_by_user"} if args.allow_paused else {"failed"}
    if state.get("phase") not in allowed_phases or state.get("pid") or state.get("training_complete"):
        raise ValueError("Only a stopped, failed training worker can be recovered")
    # Inspect in a short-lived process before launch; do not retain torch in this
    # supervisor while six IsaacSim workers compete for host RAM.
    import subprocess
    inspect_code = (
        "import torch,sys,json; from pathlib import Path; "
        "from scripts.run_three_group_experiment import preflight; "
        "cfg=json.loads(Path(sys.argv[3]).read_text()); "
        "assert preflight(cfg)==json.loads((Path(cfg['work_root'])/'input_sha256.json').read_text()); "
        "d=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "assert d['iter']==int(sys.argv[2]); "
        "assert all(k in d for k in ('actor_model_state_dict','critic_model_state_dict',"
        "'actor_optimizer_state_dict','critic_optimizer_state_dict')); "
        "print('Checkpoint iteration and optimizer states verified')"
    )
    subprocess.run([sys.executable, "-c", inspect_code, str(checkpoint),
                    str(args.completed_iteration), str(args.config.resolve())], check=True, cwd=ROOT)
    run_id = uuid.uuid4().hex[:8]
    command = recovery_command(cfg, args.group, work, checkpoint, args.completed_iteration, run_id)
    if args.check_only:
        print(json.dumps({"preflight": "passed", "gpu": cfg["gpu"],
                          "command": list(map(str, command))}, indent=2))
        return
    old_id = state["train_run_id"]
    backup = work / f"failed_state_{old_id}_{int(time.time())}.json"
    save_json(backup, state)
    recovery = {
        "started": now(), "supervisor_pid": os.getpid(), "source_run_id": old_id,
        "reason": args.reason, "source_phase": state["phase"],
        "new_run_id": run_id, "checkpoint": str(checkpoint),
        "completed_iteration": args.completed_iteration,
        "original_gpu": original_gpu,
        "runtime_gpu": cfg["gpu"],
        "first_resumed_iteration": args.completed_iteration + 1,
        "target_final_iteration": cfg["iterations"] - 1,
        "save_interval": 500,
        "allocator": {k: os.environ.get(k) for k in (
            "MALLOC_ARENA_MAX", "MALLOC_TRIM_THRESHOLD_", "HOLOSOMA_CPU_MALLOC_TRIM_INTERVAL")},
        "limitation": "Simulator/RNG/sampler state is not present in the source checkpoint; environments reinitialize.",
    }
    save_json(work / "recovery.json", recovery)
    state.update(train_run_id=run_id, recovery=recovery)
    for key in ("error", "traceback", "training_exit_code"):
        state.pop(key, None)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(cfg["gpu"]))
    env["WANDB_TAGS"] = ",".join(filter(None, [env.get("WANDB_TAGS"), "recovered", f"from-{old_id}"]))
    env["WANDB_NOTES"] = (env.get("WANDB_NOTES", "") + "\n" +
        f"Recovery of {old_id}: {args.reason}. "
        f"Resumes model/optimizers/normalizers after iteration {args.completed_iteration}; "
        "simulator and sampler reinitialize. Same reference/reward/PPO/Paper-DR. "
        "New run preserves history without overwriting previously logged steps."
    )
    try:
        run_process(command, work / "train.log", state, state_path, "training", env)
        matches = list((work / "train").rglob(f"model_{cfg['iterations'] - 1:05d}.pt"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one final checkpoint, found {matches}")
        state.update(training_complete=True, checkpoint=str(matches[0].resolve()), training_finished=now())
        save_json(state_path, state)
    except Exception as error:
        state.update(phase="failed", error=str(error), updated=now())
        save_json(state_path, state)
        raise
    result = worker(cfg, args.config.resolve(), args.group)
    if result.get("phase") != "complete":
        raise RuntimeError("Recovered training finished but postprocessing needs attention")
    if args.skip_finalization:
        return
    # The original supervisor retains this group's failed Future. Once siblings
    # finish, let it exit, then re-enter its idempotent finalization path.
    while True:
        states = [json.loads((root / f"group{g}/state.json").read_text()) for g in (1, 2, 3)]
        if any(s.get("phase") == "failed" for s in states):
            raise RuntimeError("Another group failed; joint evaluation summary deferred")
        if all(s.get("phase") == "complete" for s in states):
            with (root / "supervisor.lock").open("a") as original_lock:
                try:
                    fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    pass
                else:
                    break
        time.sleep(30)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_three_group_experiment.py"),
                    "--config", str(args.config.resolve())], check=True)


if __name__ == "__main__":
    main()
