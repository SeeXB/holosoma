"""Move local experiment inputs into data roots, retaining historical aliases.

Dry-run by default. --apply renames files on the same filesystem, never copies
external datasets, and records an auditable map under exp/data_layout/.
Run from any directory; paths are relative to this repository.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RETARGET = Path("src/holosoma_retargeting/holosoma_retargeting/demo_data")
TRAIN = Path("src/holosoma/holosoma/data")


def relocation_plan(root=ROOT):
    moves = []

    def add(source, destination):
        source, destination = Path(source), Path(destination)
        if (root / source).exists() and not (root / source).is_symlink():
            moves.append((source, destination))

    add("third_party/holosoma_upstream_audit/official_inputs", RETARGET / "omomo/official_inputs")
    add("src/holosoma_retargeting/holosoma_retargeting/models", RETARGET / "models")
    add("third_party/CARI4D/data/smpl", RETARGET / "body_models/cari4d_smpl")
    for task in sorted((root / "exp/omomo_cari4d").glob("sub*")):
        if not task.is_dir():
            continue
        target = RETARGET / "omomo/bundles" / task.name
        add(task.relative_to(root) / "input", target / "input")
        for plan in (task / "semantic_keyframes").glob("*.json"):
            add(plan.relative_to(root), target / "semantic_keyframes" / plan.name)
        for video in (task / "cari4d_friendly").glob("*_rerender.mp4"):
            add(video.relative_to(root), target / "videos" / video.name)
        for name in ("weights", "hf_cache", "sapiens_checkpoint", "hy3d_model_cache"):
            add(task.relative_to(root) / name, RETARGET / "pretrained" / task.name / name)

    for directory, dirs, _ in os.walk(root / "exp/retargeting", followlinks=False):
        for name in list(dirs):
            path = Path(directory) / name
            if path.is_symlink():
                continue
            relative = path.relative_to(root / "exp/retargeting")
            if name in ("input", "assets"):
                add(path.relative_to(root), RETARGET / "retarget_inputs" / relative)
                dirs.remove(name)
            elif name == "motions":
                add(path.relative_to(root), TRAIN / "motions/retargeted" / relative)
                dirs.remove(name)

    for task in sorted((root / "exp/training").glob("sub*")):
        add(task.relative_to(root) / "motions", TRAIN / "motions/tasks" / task.name)
        add(task.relative_to(root) / "semantic", TRAIN / "semantic" / task.name)
        for plan in task.glob("*/semantic_plan.json"):
            add(plan.relative_to(root), RETARGET / "semantic_keyframes/frozen" / task.name / plan.parent.name / plan.name)
    for directory in sorted((root / "exp").glob("benchmark*/rl")):
        for motion in directory.glob("*.npz"):
            add(motion.relative_to(root), TRAIN / "motions/benchmarks" / directory.parent.name / motion.name)
    return moves


def remap(path, moves, root=ROOT):
    path = Path(path)
    for old, new in sorted(moves, key=lambda pair: len(str(pair[0])), reverse=True):
        try:
            return root / new / path.relative_to(root / old)
        except ValueError:
            pass
    return path


def inventory(path):
    """Record inode/size/mtime without hashing multi-GB weights or following links."""
    rows = {}
    files = [path] if not path.is_dir() else [
        Path(d) / n for d, _, names in os.walk(path, followlinks=False) for n in names
    ]
    for f in files:
        if not f.is_symlink():
            stat = f.stat()
            key = "." if f == path else str(f.relative_to(path))
            rows[key] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]
    return rows


def migrate(moves, root=ROOT):
    # Check every destination before the first mutation.
    for old, new in moves:
        if os.path.lexists(root / new):
            raise FileExistsError(root / new)
    records = []
    for old, new in moves:
        source, destination = root / old, root / new
        before = inventory(source)
        relative_links = []
        if source.is_dir():
            for directory, dirs, files in os.walk(source, followlinks=False):
                for name in dirs + files:
                    link = Path(directory) / name
                    if link.is_symlink():
                        relative_links.append((link.relative_to(source), link.resolve()))
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        try:
            source.symlink_to(os.path.relpath(destination, source.parent), target_is_directory=destination.is_dir())
        except Exception:
            destination.rename(source)
            raise
        for relative, original_target in relative_links:
            link = destination / relative
            target = remap(original_target, moves, root)
            if link.resolve() != target:
                link.unlink()
                try:
                    target.relative_to(root)
                    link.symlink_to(os.path.relpath(target, link.parent))
                except ValueError:
                    link.symlink_to(target)
        if before != inventory(destination):
            raise RuntimeError(f"File identity/size/mtime changed during migration: {old}")
        records.append({"old": str(old), "new": str(new), "files": len(before),
                        "bytes": sum(row[2] for row in before.values()), "verified_identity": True})
        print(f"{old} -> {new}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--manifest", type=Path, default=ROOT / "exp/data_layout/migration_20260919.json")
    args = parser.parse_args()
    moves = relocation_plan()
    if not args.apply:
        print(json.dumps([{"old": str(a), "new": str(b)} for a, b in moves], indent=2))
        return
    if args.manifest.exists():
        raise FileExistsError(f"Preserve the previous migration manifest: {args.manifest}")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    # Save the planned map first so interrupted runs remain recoverable.
    report = {"started": datetime.now().astimezone().isoformat(), "status": "in_progress",
              "planned": [{"old": str(a), "new": str(b)} for a, b in moves]}
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")
    report["moves"] = migrate(moves)
    external = ROOT / RETARGET / "external/omomo"
    target = Path("/mnt/sdadrive/shixiongbo/omomo")
    if target.is_dir() and not os.path.lexists(external):
        external.parent.mkdir(parents=True, exist_ok=True)
        external.symlink_to(target, target_is_directory=True)
    report.update(status="complete", completed=datetime.now().astimezone().isoformat())
    args.manifest.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
