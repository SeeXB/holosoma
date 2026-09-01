#!/usr/bin/env python3
"""Create a no-copy CARI4D-compatible sequence-name view of staged inputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py


def replace_symlink(link: Path, target: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(f"Refusing to replace non-symlink: {link}")
    link.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--source-sequence", required=True)
    parser.add_argument("--alias-sequence", required=True)
    parser.add_argument("--frame-index", type=int, required=True)
    args = parser.parse_args()

    root = args.staging_root.resolve()
    videos = root / "videos"
    masks = root / "masks"
    meshes = root / "meshes"

    linked = []
    for suffix in (".0.color.mp4", ".0.color.pkl", ".0.depth-reg.mp4"):
        source = videos / f"{args.source_sequence}{suffix}"
        if source.exists():
            alias = videos / f"{args.alias_sequence}{suffix}"
            replace_symlink(alias, source)
            linked.append({"alias": str(alias), "target": str(source.resolve())})

    source_h5 = masks / f"{args.source_sequence}_masks_k0.h5"
    alias_h5 = masks / f"{args.alias_sequence}_masks_k0.h5"
    if alias_h5.is_symlink():
        alias_h5.unlink()
    elif alias_h5.exists():
        alias_h5.unlink()
    with h5py.File(alias_h5, "w") as handle:
        handle[args.alias_sequence] = h5py.ExternalLink(
            str(source_h5.resolve()), f"/{args.source_sequence}"
        )

    source_mesh_dir = meshes / f"{args.source_sequence}_{args.frame_index:03d}_rgba"
    alias_mesh_dir = meshes / f"{args.alias_sequence}_{args.frame_index:03d}_rgba"
    alias_mesh_dir.mkdir(parents=True, exist_ok=True)
    source_obj = source_mesh_dir / f"{args.source_sequence}_{args.frame_index:03d}_align.obj"
    alias_obj = alias_mesh_dir / f"{args.alias_sequence}_{args.frame_index:03d}_align.obj"
    replace_symlink(alias_obj, source_obj)

    with h5py.File(alias_h5, "r") as handle:
        group = handle[args.alias_sequence]
        dataset_count = len(group.keys())
    report = {
        "schema": "holosoma.cari4d_name_alias.v1",
        "reason": "official CARI4D parses subject from token 2 and object from token 3",
        "source_sequence": args.source_sequence,
        "alias_sequence": args.alias_sequence,
        "motion_or_pixels_modified": False,
        "video_links": linked,
        "mask_h5": str(alias_h5),
        "mask_external_link_target": str(source_h5.resolve()),
        "mask_dataset_count": dataset_count,
        "mesh_link": str(alias_obj),
        "mesh_target": str(source_obj.resolve()),
    }
    report_path = root / "sequence_name_alias.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
