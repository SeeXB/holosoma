"""G1 anatomy-to-contact-body mapping, independent of the retargeting package.

Exemptions are the task-wide union of semantic body parts, intersected with
actual simulator bodies. Only the caller's existing penalty list is reduced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

G1_BODY_PART_LINKS = {
    "pelvis": ("pelvis", "pelvis_contour_link"),
    "waist": ("waist_yaw_link", "waist_roll_link", "torso_link"),
}
for _side in ("left", "right"):
    for _part, _axes in (
        ("hip", ("pitch", "roll", "yaw")),
        ("shoulder", ("pitch", "roll", "yaw")),
        ("wrist", ("roll", "pitch", "yaw")),
        ("ankle", ("pitch", "roll")),
    ):
        G1_BODY_PART_LINKS[f"{_side}_{_part}"] = tuple(f"{_side}_{_part}_{axis}_link" for axis in _axes)
    G1_BODY_PART_LINKS[f"{_side}_knee"] = (f"{_side}_knee_link",)
    G1_BODY_PART_LINKS[f"{_side}_elbow"] = (f"{_side}_elbow_link",)
    G1_BODY_PART_LINKS[f"{_side}_hand"] = (
        f"{_side}_sphere_hand_link",
        f"{_side}_rubber_hand",
        f"{_side}_rubber_hand_link",
    )
    G1_BODY_PART_LINKS[f"{_side}_foot"] = (
        f"{_side}_foot_contact_point",
        f"{_side}_ankle_roll_link",
        "LL_FOOT" if _side == "left" else "LR_FOOT",
    )

ALIASES = {"torso": "waist", "left_forearm": "left_elbow", "right_forearm": "right_elbow"}


def load_semantic_contact_parts(path: str | Path) -> tuple[str, ...]:
    """Accept resolved plans or declarative event plans; reject misspellings."""
    from holosoma.utils.path import resolve_data_file_path

    payload = json.loads(Path(resolve_data_file_path(str(path))).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("semantic contact plan must be a JSON object")
    events = payload.get("events", payload.get("actions"))
    if not isinstance(events, list) or not events:
        raise ValueError("semantic contact exemptions require a nonempty events/actions list")
    parts = set()
    for event in events:
        names = event.get("body_parts") if isinstance(event, dict) else None
        if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
            raise ValueError("each semantic event requires nonempty string body_parts")
        parts.update(ALIASES.get(name, name) for name in names)
    unknown = parts - G1_BODY_PART_LINKS.keys()
    if unknown:
        raise ValueError(f"unknown G1 semantic contact body parts: {sorted(unknown)}")
    return tuple(sorted(parts))


def resolve_semantic_contact_links(parts: Iterable[str], body_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Resolve authored links, with a wrist-yaw fallback for fixed merged hands.

    A present fixed hand gets its own exemption. Its parent wrist is exempted
    only when the simulator has merged the hand into that parent.
    """
    available = set(body_names)
    resolved = {}
    for original in parts:
        part = ALIASES.get(original, original)
        if part not in G1_BODY_PART_LINKS:
            raise ValueError(f"unknown G1 semantic contact body part: {original!r}")
        links = tuple(name for name in G1_BODY_PART_LINKS[part] if name in available)
        if not links and part in ("left_hand", "right_hand"):
            fallback = part.replace("_hand", "_wrist_yaw_link")
            links = (fallback,) if fallback in available else ()
        if not links:
            raise ValueError(f"G1 semantic body part {part!r} has no link in simulator body_names")
        resolved[part] = links
    return resolved
