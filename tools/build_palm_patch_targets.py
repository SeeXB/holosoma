#!/usr/bin/env python3
"""Add palm position and normal targets; never use SMPL-H finger poses."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))
from holosoma_retargeting.semantic_keyframes.palm_contacts import build_palm_patch_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("contacts-path", "bundle", "body-model", "target-object-mesh", "robot-asset-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-npz", type=Path)
    parser.add_argument("--baseline-scene", type=Path)
    args = parser.parse_args()
    result = build_palm_patch_targets(**vars(args))
    print(json.dumps({k: result[k] for k in ("constraint_mode", "orientation_policy", "finger_pose_or_heading_used", "palm_geometry")}, indent=2))


if __name__ == "__main__":
    main()
