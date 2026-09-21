"""Offline reference/actual video of env 0's first episode (no new simulation)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("LP_NUM_THREADS", "2")
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diagnostics.object_drift.render_actual_vs_reference import RawVideoEncoder, font, render_qpos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=ROOT / "src/holosoma_retargeting/holosoma_retargeting/demo_data/retarget_inputs/omomo_batch/input/scenes/g1_29dof_w_clothesstand.xml")
    parser.add_argument("--frame-stride", type=int, default=1, help="Optional output downsampling, preserving playback speed")
    args = parser.parse_args()
    with np.load(args.actual, allow_pickle=False) as z, np.load(args.reference, allow_pickle=False) as ref:
        md = json.loads(str(z["_metadata_json"]))
        diagnostic = md.get("object_state_writes") == 1
        fps = float(md["fps"])
        ends = np.flatnonzero(z["done"])
        stop = int(ends[0]) + 1 if len(ends) else len(z["motion_step"])
        steps = z["motion_step"][:stop].astype(int)
        if len(steps) > 1 and steps[-1] == 0:
            stop -= 1
            steps = steps[:-1]
        order = [list(md["dof_names"]).index(str(name)) for name in ref["joint_names"]]
        actual = np.concatenate((z["pre_root_pos"][:stop], z["pre_root_quat_xyzw"][:stop, [3, 0, 1, 2]],
                                 z["pre_dof_pos"][:stop, order], z["object_pos_w"][:stop],
                                 z["object_quat_xyzw"][:stop, [3, 0, 1, 2]]), axis=1)
        reference = np.concatenate((ref["joint_pos"][steps, :36], ref["object_pos_w"][steps], ref["object_quat_w"][steps]), axis=1)
        offsets = z["reference_object_pos_w"][:stop] - ref["object_pos_w"][steps]
        reference[:, :3] += offsets
        reference[:, 36:39] += offsets
        pos_errors = z["object_pos_error_m"][:stop]
        ori_errors = np.rad2deg(z["object_ori_error_rad"][:stop])
        reasons = [name for name in ("bad_ref_pos", "bad_ref_ori", "bad_motion_body_pos", "bad_object_pos", "bad_object_ori") if bool(z[name][stop - 1])]
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    if model.nq != actual.shape[1]:
        raise ValueError("Scene/recording joint count differs")
    width, height = 640, 480
    model.vis.global_.offwidth, model.vis.global_.offheight = width, height
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation, camera.distance = 135.0, -15.0, 4.6
    camera.lookat[:] = np.mean(np.concatenate((reference[:, :3], reference[:, 36:39])), axis=0)
    camera.lookat[2] = 0.9
    renderer = mujoco.Renderer(model, height=height, width=width)
    # Half speed is explicit in the caption. Hold the last recorded state for
    # one second so very early failures are visible; no post-failure physics.
    if args.frame_stride < 1:
        raise ValueError("frame stride must be positive")
    output_fps = fps / 2 / args.frame_stride
    encoder = RawVideoEncoder(args.output, width * 2, height, output_fps)
    panels = []
    sample_indices = set(np.linspace(0, len(steps) - 1, 4, dtype=int))
    try:
        for i, (a, r) in enumerate(zip(actual, reference)):
            if i % args.frame_stride and i != len(steps) - 1 and i not in sample_indices:
                continue
            images = []
            for qpos, title in ((r, "Reference"), (a, args.label)):
                im = Image.fromarray(render_qpos(renderer, model, data, qpos, camera))
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, width, 69), fill=(22, 27, 35))
                draw.text((14, 9), title, font=font(20, True), fill="white")
                context = "no RL / full clip" if diagnostic else "env 0 / first episode"
                draw.text((14, 39), f"{context} / frame {steps[i]} / 0.5x", font=font(16), fill="white")
                if title != "Reference":
                    draw.rectangle((0, height - 60, width, height), fill=(22, 27, 35))
                    draw.text((14, height - 54), f"Object: {pos_errors[i]:.3f} m / {ori_errors[i]:.1f} deg", font=font(17), fill="white")
                    if i == len(steps) - 1:
                        end = "diagnostic end; no resets" if diagnostic else (", ".join(reasons) or "clip timeout")
                        draw.text((14, height - 29), "End: " + end, font=font(15), fill=(255, 170, 110))
                images.append(np.asarray(im))
            pair = np.concatenate(images, axis=1)
            if i % args.frame_stride == 0 or i == len(steps) - 1:
                encoder.write(pair)
            if i in sample_indices:
                panels.append(pair.copy())
        for _ in range(round(output_fps)):
            encoder.write(pair)
    finally:
        renderer.close()
        encoder.close()
    Image.fromarray(np.concatenate(panels, axis=0)).save(args.output.with_suffix(".jpg"), quality=90)
    print(f"{args.output}: first episode, {len(steps)} frames, terminal reasons={reasons}")


if __name__ == "__main__":
    main()
