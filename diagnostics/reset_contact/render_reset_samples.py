"""Render reference-vs-training-reset contact sheets from captured samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "diagnostics/reset_contact/training_style_reset_samples.npz"
DEFAULT_OUTPUT = ROOT / "diagnostics/reset_contact/rendered"
DEFAULT_MODEL = (
    ROOT
    / "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml"
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def xyzw_to_wxyz(value: np.ndarray) -> np.ndarray:
    return value[..., [3, 0, 1, 2]]


def qpos_from_sample(data: np.lib.npyio.NpzFile, index: int, reference: bool) -> np.ndarray:
    prefix = "reference_" if reference else ""
    qpos = np.empty(43, dtype=np.float64)
    qpos[:3] = data[f"{prefix}root_pos"][index]
    qpos[3:7] = xyzw_to_wxyz(data[f"{prefix}root_quat_xyzw"][index])
    qpos[7:36] = data[f"{prefix}dof_pos"][index]
    qpos[36:39] = data[f"{prefix}object_pos"][index]
    qpos[39:43] = xyzw_to_wxyz(data[f"{prefix}object_quat_xyzw"][index])
    return qpos


def make_camera(azimuth: float, elevation: float, distance: float, lookat: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance
    camera.lookat[:] = lookat
    return camera


def render(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    sim_data: mujoco.MjData,
    qpos: np.ndarray,
    camera: mujoco.MjvCamera,
) -> np.ndarray:
    sim_data.qpos[:] = qpos
    sim_data.qvel[:] = 0.0
    mujoco.mj_forward(model, sim_data)
    renderer.update_scene(sim_data, camera=camera)
    return renderer.render().copy()


def annotate(image: np.ndarray, title: str, lines: list[str], color: tuple[int, int, int]) -> Image.Image:
    result = Image.fromarray(image)
    draw = ImageDraw.Draw(result, "RGBA")
    width, _ = result.size
    height = 42 + 25 * len(lines)
    draw.rounded_rectangle((12, 10, width - 12, 10 + height), 8, fill=(7, 11, 17, 218))
    draw.rectangle((12, 10, 22, 10 + height), fill=(*color, 255))
    draw.text((34, 18), title, font=font(21, bold=True), fill=(250, 250, 250, 255))
    for row, line in enumerate(lines):
        draw.text((34, 51 + row * 25), line, font=font(16), fill=(235, 239, 244, 255))
    return result


def body_positions(model: mujoco.MjModel, sim_data: mujoco.MjData, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim_data.qpos[:] = qpos
    sim_data.qvel[:] = 0.0
    mujoco.mj_forward(model, sim_data)
    left = sim_data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand_link")].copy()
    right = sim_data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand_link")].copy()
    return left, right


def hand_box_surface_distances(
    model: mujoco.MjModel,
    sim_data: mujoco.MjData,
    qpos: np.ndarray,
) -> np.ndarray:
    """Signed mesh distances for left/right rubber hand to the box."""

    sim_data.qpos[:] = qpos
    sim_data.qvel[:] = 0.0
    mujoco.mj_forward(model, sim_data)
    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "largebox")
    hands = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_rubber_hand_link"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_rubber_hand_link"),
    ]
    return np.array([mujoco.mj_geomDistance(model, sim_data, hand, box, 1.0, None) for hand in hands])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.input.resolve(), allow_pickle=False)
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    if model.nq != 43:
        raise ValueError(f"expected model nq=43, got {model.nq}")
    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    sim_data = mujoco.MjData(model)

    steps = data["motion_step"].astype(np.int64)
    ref_object = data["reference_object_pos"]
    actual_object = data["object_pos"]
    root_delta = data["root_pos"] - data["reference_root_pos"]
    object_delta = actual_object - ref_object
    relative_delta = object_delta - root_delta

    # Restrict the visual diagnosis to the portion where the reference box is
    # lifted.  Rank by the independent robot/object XY displacement introduced
    # by reset, while keeping selected frames spread across the motion.
    ground_z = float(np.median(ref_object[steps < 20, 2])) if np.any(steps < 20) else float(np.min(ref_object[:, 2]))
    candidates = np.where(ref_object[:, 2] > ground_z + 0.08)[0]
    scores = np.linalg.norm(relative_delta[:, :2], axis=1)
    ranked = candidates[np.argsort(scores[candidates])[::-1]]
    selected: list[int] = []
    for index in ranked:
        if all(abs(int(steps[index]) - int(steps[old])) >= 12 for old in selected):
            selected.append(int(index))
        if len(selected) >= args.count:
            break

    reference_surface_distance = np.empty((len(steps), 2), dtype=np.float64)
    reset_surface_distance = np.empty((len(steps), 2), dtype=np.float64)
    for index in range(len(steps)):
        reference_surface_distance[index] = hand_box_surface_distances(
            model, sim_data, qpos_from_sample(data, index, reference=True)
        )
        reset_surface_distance[index] = hand_box_surface_distances(
            model, sim_data, qpos_from_sample(data, index, reference=False)
        )

    metrics: list[dict[str, object]] = []
    sheets: list[Image.Image] = []
    try:
        for rank, index in enumerate(selected, start=1):
            actual_qpos = qpos_from_sample(data, index, reference=False)
            reference_qpos = qpos_from_sample(data, index, reference=True)
            ref_left, ref_right = body_positions(model, sim_data, reference_qpos)
            act_left, act_right = body_positions(model, sim_data, actual_qpos)
            ref_center = ref_object[index]
            act_center = actual_object[index]
            ref_dist = np.array([np.linalg.norm(ref_left - ref_center), np.linalg.norm(ref_right - ref_center)])
            act_dist = np.array([np.linalg.norm(act_left - act_center), np.linalg.norm(act_right - act_center)])
            ref_surface = reference_surface_distance[index]
            act_surface = reset_surface_distance[index]
            lookat = 0.5 * (ref_center + data["reference_root_pos"][index])
            lookat[2] = max(0.55, float(ref_center[2]))
            side_camera = make_camera(142.0, -16.0, 2.8, lookat)
            top_camera = make_camera(90.0, -72.0, 3.0, lookat)

            line1 = f"motion frame {steps[index]:03d}   sample {index}"
            line2 = f"relative reset offset XY: {relative_delta[index, 0] * 100:+.1f}, {relative_delta[index, 1] * 100:+.1f} cm"
            ref_line3 = f"hand-center L/R: {ref_dist[0] * 100:.1f}/{ref_dist[1] * 100:.1f} cm"
            act_line3 = f"hand-center L/R: {act_dist[0] * 100:.1f}/{act_dist[1] * 100:.1f} cm"
            ref_line4 = f"hand-box gap L/R: {ref_surface[0] * 100:+.1f}/{ref_surface[1] * 100:+.1f} cm"
            act_line4 = f"hand-box gap L/R: {act_surface[0] * 100:+.1f}/{act_surface[1] * 100:+.1f} cm"

            ref_side = annotate(
                render(renderer, model, sim_data, reference_qpos, side_camera),
                "Reference at sampled phase",
                [line1, "no initialization noise", ref_line3, ref_line4],
                (39, 155, 93),
            )
            act_side = annotate(
                render(renderer, model, sim_data, actual_qpos, side_camera),
                "Training-style reset",
                [line1, line2, act_line3, act_line4],
                (220, 71, 71),
            )
            ref_top = annotate(
                render(renderer, model, sim_data, reference_qpos, top_camera),
                "Reference — top view",
                [line1, ref_line3, ref_line4],
                (39, 155, 93),
            )
            act_top = annotate(
                render(renderer, model, sim_data, actual_qpos, top_camera),
                "Reset — top view",
                [line1, line2, act_line3, act_line4],
                (220, 71, 71),
            )
            sheet = Image.new("RGB", (2 * args.width, 2 * args.height), (12, 15, 20))
            sheet.paste(ref_side, (0, 0))
            sheet.paste(act_side, (args.width, 0))
            sheet.paste(ref_top, (0, args.height))
            sheet.paste(act_top, (args.width, args.height))
            output = args.output_dir / f"reset_sample_{rank:02d}_frame_{steps[index]:03d}.png"
            sheet.save(output)
            sheets.append(sheet)
            metrics.append(
                {
                    "rank": rank,
                    "sample_index": index,
                    "motion_frame": int(steps[index]),
                    "relative_reset_offset_xyz_m": relative_delta[index].tolist(),
                    "reference_hand_center_distance_m": ref_dist.tolist(),
                    "reset_hand_center_distance_m": act_dist.tolist(),
                    "reference_hand_box_signed_distance_m": ref_surface.tolist(),
                    "reset_hand_box_signed_distance_m": act_surface.tolist(),
                    "image": str(output),
                }
            )
    finally:
        renderer.close()

    if sheets:
        thumb_width = 480
        thumb_height = 480
        columns = 2
        rows = (len(sheets) + columns - 1) // columns
        contact_sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (16, 19, 24))
        for i, sheet in enumerate(sheets):
            thumb = sheet.copy()
            thumb.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
            contact_sheet.paste(thumb, ((i % columns) * thumb_width, (i // columns) * thumb_height))
        contact_sheet.save(args.output_dir / "reset_samples_contact_sheet.png")

    summary = {
        "input": str(args.input.resolve()),
        "sample_count": int(len(steps)),
        "lifted_candidate_count": int(len(candidates)),
        "selected_count": len(selected),
        "all_samples_relative_xy_offset_cm": {
            "mean": float(np.mean(np.linalg.norm(relative_delta[:, :2], axis=1)) * 100),
            "p50": float(np.quantile(np.linalg.norm(relative_delta[:, :2], axis=1), 0.5) * 100),
            "p90": float(np.quantile(np.linalg.norm(relative_delta[:, :2], axis=1), 0.9) * 100),
            "max": float(np.max(np.linalg.norm(relative_delta[:, :2], axis=1)) * 100),
        },
        "lifted_samples_hand_box_gap": {
            "candidate_count": int(np.sum(ref_object[:, 2] > ground_z + 0.08)),
            "either_hand_gap_gt_1cm_percent": float(
                100
                * np.mean(
                    np.any(reset_surface_distance[ref_object[:, 2] > ground_z + 0.08] > 0.01, axis=1)
                )
            ),
            "either_hand_gap_gt_2cm_percent": float(
                100
                * np.mean(
                    np.any(reset_surface_distance[ref_object[:, 2] > ground_z + 0.08] > 0.02, axis=1)
                )
            ),
            "both_hands_gap_gt_2cm_percent": float(
                100
                * np.mean(
                    np.all(reset_surface_distance[ref_object[:, 2] > ground_z + 0.08] > 0.02, axis=1)
                )
            ),
            "max_positive_gap_cm": float(
                100 * np.max(reset_surface_distance[ref_object[:, 2] > ground_z + 0.08])
            ),
            "min_penetration_cm": float(
                100 * np.min(reset_surface_distance[ref_object[:, 2] > ground_z + 0.08])
            ),
        },
        "selected": metrics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
