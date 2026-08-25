"""Dependency-light helpers shared by the HOI export and validation tools."""

from __future__ import annotations

import json
import math
import re
import shlex
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    """Convert NumPy/path values into JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_obj_vertices(path: Path) -> np.ndarray:
    """Read OBJ vertex positions without requiring trimesh."""
    vertices: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed OBJ vertex at {path}:{line_number}: {line.rstrip()}")
            vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    if not vertices:
        raise ValueError(f"OBJ contains no vertices: {path}")
    return np.asarray(vertices, dtype=np.float64)


def count_obj_faces(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        return sum(line.startswith("f ") for line in stream)


def obj_extent(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = read_obj_vertices(path)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return lower, upper, upper - lower


def _copy_obj_materials(source: Path, destination_dir: Path) -> None:
    """Copy local MTL/texture sidecars referenced by an OBJ or fail explicitly."""
    material_files: list[tuple[Path, Path]] = []
    with source.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("mtllib "):
                for token in shlex.split(line)[1:]:
                    reference = Path(token)
                    if reference.is_absolute():
                        raise ValueError(f"OBJ uses a non-portable absolute material path: {reference}")
                    material_files.append((source.parent / reference, reference))
    output_root = destination_dir.resolve()
    for material, material_reference in material_files:
        if not material.is_file():
            raise FileNotFoundError(f"OBJ references a missing material file: {material}")
        target = destination_dir / material_reference
        if not target.resolve().is_relative_to(output_root):
            raise ValueError(f"OBJ material path escapes the output directory: {material_reference}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if material.resolve() != target.resolve():
            shutil.copy2(material, target)
        with material.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                fields = shlex.split(line)
                if len(fields) < 2 or not fields[0].lower().startswith("map_"):
                    continue
                texture_reference = Path(fields[-1])
                if texture_reference.is_absolute():
                    raise ValueError(f"MTL uses a non-portable absolute texture path: {texture_reference}")
                texture = material.parent / texture_reference
                if not texture.is_file():
                    raise FileNotFoundError(f"MTL references a missing texture file: {texture}")
                texture_target = target.parent / texture_reference
                if not texture_target.resolve().is_relative_to(output_root):
                    raise ValueError(f"MTL texture path escapes the output directory: {texture_reference}")
                texture_target.parent.mkdir(parents=True, exist_ok=True)
                if texture.resolve() != texture_target.resolve():
                    shutil.copy2(texture, texture_target)


def copy_obj_with_materials(source: Path, destination: Path) -> None:
    """Byte-copy an OBJ and local material sidecars without modifying its geometry."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    _copy_obj_materials(source, destination.parent)


def transform_obj_vertices(
    source: Path,
    destination: Path,
    *,
    scale: np.ndarray | float = 1.0,
    offset: np.ndarray | None = None,
    sanitize_invalid_face_normals: bool = False,
) -> int:
    """Copy an OBJ while applying ``v_out = scale * v_in + offset``."""
    scale_array = np.asarray(scale, dtype=np.float64)
    if scale_array.ndim == 0:
        scale_array = np.repeat(scale_array, 3)
    if scale_array.shape != (3,):
        raise ValueError(f"OBJ scale must be scalar or length 3, got {scale_array.shape}")
    offset_array = np.zeros(3, dtype=np.float64) if offset is None else np.asarray(offset, dtype=np.float64)
    if offset_array.shape != (3,):
        raise ValueError(f"OBJ offset must have shape (3,), got {offset_array.shape}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as input_stream:
        lines = input_stream.readlines()
    sanitized_faces = 0
    with destination.open("w", encoding="utf-8") as output_stream:
        for line_number, line in enumerate(lines, start=1):
            if sanitize_invalid_face_normals and line.startswith("f "):
                tokens = line.split()
                components = [token.split("/") for token in tokens[1:]]
                if any(parts[0] == "0" or (len(parts) >= 2 and parts[1] == "0") for parts in components):
                    raise ValueError(
                        f"OBJ face at {source}:{line_number} uses invalid zero vertex/UV indices; "
                        "repair would change topology or visual correspondence"
                    )
                if any(len(parts) >= 3 and parts[2] == "0" for parts in components):
                    repaired = []
                    for parts in components:
                        repaired.append(parts[0] if len(parts) < 2 or not parts[1] else f"{parts[0]}/{parts[1]}")
                    output_stream.write("f " + " ".join(repaired) + "\n")
                    sanitized_faces += 1
                    continue
            if not line.startswith("v "):
                output_stream.write(line)
                continue
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed OBJ vertex at {source}:{line_number}: {line.rstrip()}")
            vertex = np.asarray([float(fields[1]), float(fields[2]), float(fields[3])], dtype=np.float64)
            transformed = scale_array * vertex + offset_array
            suffix = "" if len(fields) == 4 else " " + " ".join(fields[4:])
            output_stream.write(
                f"v {transformed[0]:.10g} {transformed[1]:.10g} {transformed[2]:.10g}{suffix}\n"
            )
    _copy_obj_materials(source, destination.parent)
    if sanitized_faces:
        print(
            f"OBJ compatibility repair: removed invalid zero normal indices from {sanitized_faces} face(s) "
            f"while preserving vertices, UVs, and topology: {destination}"
        )
    return sanitized_faces


def matrix_to_quaternion_wxyz(matrices: np.ndarray) -> np.ndarray:
    """Convert proper rotation matrices to normalized scalar-first quaternions."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotations with shape [T,3,3], got {matrices.shape}")
    quaternions = np.empty((len(matrices), 4), dtype=np.float64)
    for index, matrix in enumerate(matrices):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            root = math.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [0.25 * root, (matrix[2, 1] - matrix[1, 2]) / root,
                 (matrix[0, 2] - matrix[2, 0]) / root, (matrix[1, 0] - matrix[0, 1]) / root]
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                root = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
                quat = np.array(
                    [(matrix[2, 1] - matrix[1, 2]) / root, 0.25 * root,
                     (matrix[0, 1] + matrix[1, 0]) / root, (matrix[0, 2] + matrix[2, 0]) / root]
                )
            elif axis == 1:
                root = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
                quat = np.array(
                    [(matrix[0, 2] - matrix[2, 0]) / root, (matrix[0, 1] + matrix[1, 0]) / root,
                     0.25 * root, (matrix[1, 2] + matrix[2, 1]) / root]
                )
            else:
                root = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
                quat = np.array(
                    [(matrix[1, 0] - matrix[0, 1]) / root, (matrix[0, 2] + matrix[2, 0]) / root,
                     (matrix[1, 2] + matrix[2, 1]) / root, 0.25 * root]
                )
        norm = np.linalg.norm(quat)
        if norm < 1e-12:
            raise ValueError(f"Could not convert rotation matrix {index} to a quaternion")
        quat /= norm
        if quat[0] < 0.0:
            quat *= -1.0
        quaternions[index] = quat
    return quaternions.astype(np.float32)


def parse_cari4d_frame_ids(frames: list[Any]) -> tuple[np.ndarray, list[str]]:
    labels = [str(frame) for frame in frames]
    identifiers: list[int] = []
    for label in labels:
        leaf = label.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        match = re.fullmatch(r"(?:frame_)?(\d+)", leaf)
        if match is None:
            raise ValueError(
                f"CARI4D frame label {label!r} has no unambiguous numeric ID; refusing to invent frame_ids"
            )
        identifiers.append(int(match.group(1)))
    result = np.asarray(identifiers, dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError("CARI4D result contains duplicate frame IDs")
    if len(result) > 1 and np.any(np.diff(result) <= 0):
        raise ValueError("CARI4D frame IDs are not strictly increasing")
    return result, labels


def find_best_scale(scale_data: dict[str, Any]) -> np.ndarray:
    """Read CARI4D's actual ``best_scale`` serialization without filename inference."""
    if "best_scale" not in scale_data:
        raise KeyError("Scale JSON does not contain CARI4D's required 'best_scale' field")
    scale = np.asarray(scale_data["best_scale"], dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(scale, 3)
    if scale.shape != (3,):
        raise ValueError(f"CARI4D best_scale must be scalar or length 3, got shape {scale.shape}")
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(f"CARI4D best_scale must be finite and positive, got {scale}")
    return scale


def relative_extent_error(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.abs(expected), 1e-12)
    return np.abs(np.asarray(actual) - np.asarray(expected)) / denominator
