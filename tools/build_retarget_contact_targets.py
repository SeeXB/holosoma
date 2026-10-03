#!/usr/bin/env python3
"""Build semantic-key-part surface-contact targets for retargeting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/holosoma_retargeting"))

from holosoma_retargeting.semantic_keyframes.contact_targets import (  # noqa: E402
    build_retarget_contact_targets,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--body-model", type=Path, required=True)
    parser.add_argument("--target-object-mesh", type=Path, required=True)
    parser.add_argument("--semantic-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--onset-threshold-m", type=float, default=0.02)
    parser.add_argument("--hold-threshold-m", type=float, default=0.03)
    parser.add_argument("--activation-mode", choices=("per_part", "semantic_group"), default="per_part")
    args = parser.parse_args()
    targets = build_retarget_contact_targets(
        bundle=args.bundle,
        body_model=args.body_model,
        target_object_mesh=args.target_object_mesh,
        semantic_plan=args.semantic_plan,
        output=args.output,
        onset_threshold_m=args.onset_threshold_m,
        hold_threshold_m=args.hold_threshold_m,
        activation_mode=args.activation_mode,
    )
    print(json.dumps(targets.metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
