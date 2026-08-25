#!/usr/bin/env python3
"""Validate visual/collision/rigid/mass/metric properties of an object USD."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--usd", required=True, type=Path)
parser.add_argument("--source-obj", required=True, type=Path)
parser.add_argument(
    "--expected-collision",
    choices=("convexDecomposition", "convexHull", "none"),
    default="convexDecomposition",
)
parser.add_argument("--expected-mass", type=float, default=1.0)
parser.add_argument("--report", type=Path)
parser.add_argument("--bbox-relative-tolerance", type=float, default=1e-2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from hoi_pipeline.common import json_ready
from hoi_pipeline.usd_validation import validate_object_usd, write_usd_report


def main() -> None:
    usd_path = args.usd.expanduser().resolve()
    source_obj = args.source_obj.expanduser().resolve()
    report = validate_object_usd(
        usd_path,
        source_obj,
        expected_collision=args.expected_collision,
        expected_mass=args.expected_mass,
        bbox_relative_tolerance=args.bbox_relative_tolerance,
    )
    report_path = args.report
    if report_path is None:
        report_path = (
            usd_path.parent.parent / "validation" / "usd_report.json"
            if usd_path.parent.name == "object"
            else usd_path.with_suffix(".validation.json")
        )
    write_usd_report(report_path.expanduser().resolve(), report)
    print(json.dumps(json_ready(report), indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
