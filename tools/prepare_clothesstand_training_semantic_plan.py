#!/usr/bin/env python3
"""Prepare the task-specific semantic plan used by the clothes-stand RL run.

The VLM event names and keyframe functions are retained from the generated
OMOMO plan.  Its resolved windows happened to collapse to the same frame for
all three actions (67), which is valid for the retargeting-side window schema
but cannot define ordered intervals for ``semantic_adaptive`` sampling.  This
small training adapter records the visible phase boundaries of this clip:
pick-up/lift (0--30), carry (30--95), and placement/release (95--125).
It does not change either retargeted trajectory.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


DEFAULT_SOURCE = Path(
    "exp/omomo_cari4d/sub9_clothesstand_058/semantic_keyframes/"
    "sub9_clothesstand_058_dynamic.json"
)
DEFAULT_OUTPUT = Path(
    "exp/training/sub9_clothesstand_058/semantic/"
    "sub9_clothesstand_058_semantic_adaptive_training.json"
)

TRIGGERS = {
    "pick_up_clothes_stand": (0, 30),
    "walk_with_clothes_stand": (30, 95),
    "place_clothes_stand": (95, 125),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = json.loads(args.source.read_text(encoding="utf-8"))
    if payload.get("fps") != 30 or not isinstance(payload.get("events"), list):
        raise ValueError("source must be a resolved 30-fps semantic event plan")

    result = copy.deepcopy(payload)
    seen: set[str] = set()
    for event in result["events"]:
        name = event.get("event")
        if name not in TRIGGERS or name in seen:
            raise ValueError(f"unexpected or duplicate clothes-stand event: {name!r}")
        seen.add(name)
        start, end = TRIGGERS[name]
        event["windows"] = [{"start_frame": start, "end_frame": end, "trigger_frame": start}]
    if seen != set(TRIGGERS):
        raise ValueError(f"missing clothes-stand events: {sorted(set(TRIGGERS) - seen)}")

    result["semantic_mode"] = "dynamic_vlm_functions_training_phase_adapter"
    metadata = dict(result.get("generation_metadata", {}))
    metadata.update(
        {
            "training_plan_adapter": "ordered_visible_phase_boundaries",
            "training_plan_source": str(args.source.resolve()),
            "training_plan_note": (
                "The VLM plan resolved all three action starts to frame 67; ordered phase boundaries "
                "0/30/95 were supplied for semantic-adaptive reset sampling only."
            ),
        }
    )
    result["generation_metadata"] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "triggers": TRIGGERS}, ensure_ascii=False))


if __name__ == "__main__":
    main()
