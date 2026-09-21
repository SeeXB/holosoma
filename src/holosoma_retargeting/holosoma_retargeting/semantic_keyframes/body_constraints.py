"""Explicit task constraints, separate from the model's visual annotations."""

from copy import deepcopy


def apply_body_part_constraints(plan):
    """Pair annotated upper-limb parts for a task explicitly marked bimanual.

    Preserve model phases and record additions; never infer bimanual interaction
    from object category, nor exempt lower-limb contacts by symmetry.
    """
    constraint = plan.get("body_part_constraints")
    if constraint is None:
        return plan
    if constraint.get("mode") != "bimanual_upper_limbs_v1":
        raise ValueError(f"Unknown body part constraint: {constraint!r}")
    result = deepcopy(plan)
    for action in result["actions"]:
        original = action.setdefault("body_parts_before_constraints", list(action["body_parts"]))
        parts = list(dict.fromkeys(original))
        for name in original:
            side, _, part = name.partition("_")
            if side in ("left", "right") and part in ("shoulder", "elbow", "wrist", "hand"):
                partner = ("right" if side == "left" else "left") + "_" + part
                if partner not in parts:
                    parts.append(partner)
        action["body_parts"] = parts
        action["body_parts_added_by_constraints"] = [p for p in parts if p not in original]
    return result
