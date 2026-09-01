from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


SEQUENCE_ALIASES = {
    "sub03_largebox3": "sub3_largebox_003",
    "sub03_largebox_003": "sub3_largebox_003",
    "sub3_largebox3": "sub3_largebox_003",
    "sub3_largebox_003": "sub3_largebox_003",
}


def canonical_sequence_name(name: str) -> str:
    try:
        return SEQUENCE_ALIASES[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown sequence {name!r}; this regression tool only accepts "
            f"{sorted(SEQUENCE_ALIASES)}"
        ) from exc


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def camera_position(
    target: np.ndarray, azimuth_deg: float, elevation_deg: float, distance: float
) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    direction = np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ],
        dtype=np.float64,
    )
    return np.asarray(target, dtype=np.float64) + distance * direction


def normalized_dimensions(dimensions: np.ndarray) -> np.ndarray:
    dimensions = np.asarray(dimensions, dtype=np.float64)
    return dimensions / max(float(dimensions.max()), 1.0e-12)
