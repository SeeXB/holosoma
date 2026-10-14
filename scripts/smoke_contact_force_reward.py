"""Run a small, isolated force-contact training smoke test without evaluation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_three_group_experiment import now, preflight, run_process, save_json, train_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    source = json.loads(args.config.read_text())
    if source.get("semantic_reward_mode") not in ("contact_force", "contact_force_shaped"):
        raise ValueError("Smoke test requires explicit force-contact mode")
    cfg = {**source, "num_envs": 64, "iterations": 6, "save_interval": 2,
           "wandb": False, "tag": source["tag"] + "_smoke"}
    preflight(cfg)
    work = args.config.resolve().parent / "smoke"
    work.mkdir(exist_ok=True)
    state_path = work / "state.json"
    if state_path.exists():
        raise FileExistsError(state_path)
    state = {"started": now(), "source_config": str(args.config.resolve()), "gpu": args.gpu}
    command = train_command(cfg, 3, work / "train", "smoke")
    command[command.index("--training.export-onnx") + 1] = "False"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    run_process(command, work / "train.log", state, state_path, "smoke", env)
    checkpoints = list((work / "train").rglob("model_00005.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError("Smoke test did not complete its final checkpoint")
    state.update(phase="passed", checkpoint=str(checkpoints[0]), updated=now())
    save_json(state_path, state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
