#!/usr/bin/env python3
"""Parameterised wrapper around CARI4D's official custom-video command sequence."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import runpy
import shlex
import subprocess
import sys
from pathlib import Path

from hoi_pipeline.common import copy_obj_with_materials, write_json


def parse_args() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    default_cari4d = repository / "third_party" / "CARI4D"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Canonical sequence output directory.")
    parser.add_argument("--cari4d-root", type=Path, default=default_cari4d)
    parser.add_argument(
        "--torch-home",
        type=Path,
        help="DINOv2 torch.hub cache (default: <cari4d-root>/.cache/torch).",
    )
    parser.add_argument(
        "--unidepth-model",
        type=Path,
        help="Local UniDepthV2 Hub snapshot containing config.json and model.safetensors.",
    )
    parser.add_argument("--masks-root", type=Path, help="Directory containing <seq>_masks_k0.h5.")
    parser.add_argument("--packed-root", type=Path, help="Directory containing <seq>_GT-packed.pkl.")
    parser.add_argument("--object-mesh-normalized", type=Path, help="Exact normalized Hunyuan3D/SAM3D OBJ.")
    parser.add_argument(
        "--upstream-provenance-json",
        type=Path,
        action="append",
        default=[],
        help="Preprocessing provenance JSON to carry into the native manifest and canonical export; repeatable.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Joint-optimization batch size.")
    parser.add_argument("--num-steps", type=int, default=3000)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--fps", type=float, help="Forwarded only if the video FPS cannot be read by exporter.")
    return parser.parse_args()


def _run(command: list[str], *, cwd: Path, dry_run: bool = False) -> None:
    print("+ " + shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def _require_file(path: Path, message: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"Missing {path}\n  Next command: {message}")


def _require_sha256(path: Path, expected: str, message: str, errors: list[str]) -> None:
    if not path.is_file():
        _require_file(path, message, errors)
        return
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        errors.append(
            f"Invalid or incomplete checkpoint {path}\n"
            f"  expected SHA256: {expected}\n"
            f"  actual SHA256:   {actual}\n"
            f"  Next command: {message}"
        )


def _require_smplx_smplh(path: Path, errors: list[str]) -> None:
    if not path.is_file():
        return
    try:
        with path.open("rb") as stream:
            model_data = pickle.load(stream, encoding="latin1")
        keys = set(model_data if isinstance(model_data, dict) else vars(model_data))
    except Exception as exc:
        errors.append(f"Cannot inspect SMPL-H model {path}: {type(exc).__name__}: {exc}")
        return
    required = {"hands_componentsl", "hands_componentsr", "hands_meanl", "hands_meanr"}
    missing = sorted(required - keys)
    if missing:
        errors.append(
            f"SMPL-H model {path} lacks fields required by the installed smplx loader: {missing}\n"
            "  Next step: use an SMPL-H pickle packaged for smplx, or merge the matching licensed "
            "MANO_LEFT/RIGHT hands_components and hands_mean fields into a separate compatibility copy"
        )


def _unique_result(root: Path, sequence: str, stage: str) -> Path:
    matches = sorted(root.glob(f"*/{sequence}.pth"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {stage} result under {root}, found {len(matches)}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0].resolve()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError(f"--batch-size and --num-steps must be positive, got {args.batch_size}, {args.num_steps}")
    repository = Path(__file__).resolve().parents[1]
    video = args.video.expanduser().resolve()
    output = args.output.expanduser().resolve()
    cari4d = args.cari4d_root.expanduser().resolve()
    torch_home = (args.torch_home or cari4d / ".cache" / "torch").expanduser().resolve()
    unidepth_model = args.unidepth_model.expanduser().resolve() if args.unidepth_model else None
    os.environ["TORCH_HOME"] = str(torch_home)
    if not video.name.endswith(".0.color.mp4"):
        raise ValueError(
            "CARI4D custom-video loaders require '<sequence>.0.color.mp4'. Rename/copy the video explicitly; "
            "the wrapper will not reinterpret its naming convention."
        )
    sequence = video.name.removesuffix(".0.color.mp4")
    if len(sequence.split("_")) < 3:
        raise ValueError(
            "Current CARI4D source indexes sequence.split('_')[1] for subject and [2] for object. "
            f"Sequence {sequence!r} does not satisfy that native requirement."
        )
    subject_token = sequence.split("_")[1]
    const_file = cari4d / "behave_data" / "const.py"
    if not const_file.is_file():
        raise FileNotFoundError(f"Invalid CARI4D checkout; missing {const_file}")
    subject_genders = runpy.run_path(str(const_file))["_sub_gender"]
    if subject_token not in subject_genders:
        raise ValueError(
            f"CARI4D's checked-out behave_data.const._sub_gender has no entry for subject token {subject_token!r}. "
            "Use a source-supported token that matches the person's gender (for example Sub01=male or "
            "Sub06=female), or explicitly extend CARI4D metadata; the wrapper will not guess gender."
        )
    subject_gender = subject_genders[subject_token]
    defaults = cari4d / "data" / "cari4d-demo" / "videogen"
    masks_root = (args.masks_root or defaults / "masks").expanduser().resolve()
    packed_root = (args.packed_root or defaults / "packed").expanduser().resolve()
    normalized_mesh = args.object_mesh_normalized.expanduser().resolve() if args.object_mesh_normalized else None
    upstream_provenance = [path.expanduser().resolve() for path in args.upstream_provenance_json]
    native = output / "cari4d_native"
    nlf_root = native / "nlf"
    fp_root = native / "fp-hy3d-track"
    coconet_root = native / "coconet"
    opt_root = native / "opt"
    staged_mesh_root = native / "normalized_mesh"
    metric_mesh_root = native / "metric_mesh"

    errors: list[str] = []
    _require_file(
        video,
        "place the RGB video at <sequence>.0.color.mp4",
        errors,
    )
    _require_file(
        masks_root / f"{sequence}_masks_k0.h5",
        f"(SAM3 environment) python prep/run_sam3_masks.py --video {video} "
        f"--human_prompt person --object_prompt '<object>' --visualize",
        errors,
    )
    _require_file(
        packed_root / f"{sequence}_GT-packed.pkl",
        f"(CARI4D/Sapiens environment) python prep/run_sapiens_pose.py --video {video} "
        f"--masks_root {masks_root} --packed_root {packed_root}",
        errors,
    )
    if normalized_mesh is None or not normalized_mesh.is_file():
        errors.append(
            "Missing explicit --object-mesh-normalized. The final CARI4D .pth does not record this path, so "
            "the wrapper refuses filename-based scale inference.\n"
            f"  Next command: (Hunyuan3D environment) python prep/run_hy3d_recon.py --video {video} "
            f"--masks_root {masks_root} --hy3d_root <mesh-root> --blender_path <blender>"
        )
    for path in upstream_provenance:
        _require_file(path, "provide an existing preprocessing provenance JSON", errors)
    if unidepth_model is not None:
        _require_file(
            unidepth_model / "config.json",
            "download the UniDepth Hub config.json into --unidepth-model",
            errors,
        )
        _require_file(
            unidepth_model / "model.safetensors",
            "download the UniDepth Hub model.safetensors into --unidepth-model",
            errors,
        )
    required_cari4d_files = {
        cari4d / "weights" / "nlf_l_multi_0.3.2.torchscript": "download the NLF v0.3.2 checkpoint (CARI4D README)",
        cari4d / "experiments" / "cari4d-release" / "step031397.pth": "download CARI4D step031397.pth",
        cari4d / "weights" / "2023-10-28-18-33-37" / "model_best.pth": "install FoundationPose refiner weights",
        cari4d / "weights" / "2023-10-28-18-33-37" / "config.yml": "install FoundationPose refiner config",
        cari4d / "weights" / "2024-01-11-20-02-45" / "model_best.pth": "install FoundationPose scorer weights",
        cari4d / "weights" / "2024-01-11-20-02-45" / "config.yml": "install FoundationPose scorer config",
        cari4d / "data" / "smpl" / f"SMPLH_{subject_gender}.pkl": (
            f"make the licensed SMPL-H {subject_gender} model visible at CARI4D's actual source path; "
            f"the README's data/smpl/smplh/ file may be symlinked to data/smpl/SMPLH_{subject_gender}.pkl"
        ),
        cari4d / "data" / "smpl" / "smplh" / f"SMPLH_{subject_gender.upper()}.pkl": (
            f"make the licensed SMPL-H {subject_gender} model visible at the uppercase filename assembled by "
            "smplx.create(model_path='data/smpl', model_type='smplh', ...); the README's lowercase file may "
            "be exposed with a symlink"
        ),
        cari4d / "data" / "assets" / "body25_regressor.pkl": "extract CARI4D demo body landmarks",
        cari4d / "data" / "assets" / "face_regressor.pkl": "extract CARI4D demo face landmarks",
        cari4d / "data" / "assets" / "hand_regressor.pkl": "extract CARI4D demo hand landmarks",
        cari4d / "data" / "assets" / "smpl_parts_dense.pkl": "extract CARI4D demo SMPL part labels",
        cari4d / "data" / "assets" / "smpl-meshes" / "parts_surrel.obj": (
            "extract CARI4D demo SMPL visualization mesh"
        ),
    }
    for path, instruction in required_cari4d_files.items():
        _require_file(path, instruction, errors)
    _require_smplx_smplh(
        cari4d / "data" / "smpl" / "smplh" / f"SMPLH_{subject_gender.upper()}.pkl",
        errors,
    )
    dino_repositories = sorted((torch_home / "hub").glob("facebookresearch_dinov2_*"))
    if not any((path / "hubconf.py").is_file() for path in dino_repositories):
        errors.append(
            f"Missing DINOv2 torch.hub source under {torch_home / 'hub'}\n"
            f"  Next command: TORCH_HOME={torch_home} python -c \"import torch; "
            "torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True)\""
        )
    dino_checkpoints = {
        "dinov2_vits14_pretrain.pth": "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9",
        "dinov2_vitb14_pretrain.pth": "0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73",
    }
    for dino_checkpoint, expected_sha256 in dino_checkpoints.items():
        _require_sha256(
            torch_home / "hub" / "checkpoints" / dino_checkpoint,
            expected_sha256,
            f"preload the official {dino_checkpoint} into TORCH_HOME={torch_home}",
            errors,
        )
    _require_file(
        torch_home / "hub" / "checkpoints" / f"VolumetricSMPL_smpl_{subject_gender}.ckpt",
        "preload the checkpoint requested by VolumetricSMPL.attach_volume() from the official "
        "markomih/VolumetricSMPL dev/models directory",
        errors,
    )
    for module_dir in (cari4d / "unidepth", cari4d / "VolumetricSMPL"):
        if not module_dir.is_dir():
            errors.append(f"Missing CARI4D dependency directory {module_dir}\n  Next step: follow CARI4D README setup")
    if errors:
        print("CARI4D preflight: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(2)
    print("CARI4D preflight: PASS")
    if args.preflight_only:
        return

    for directory in (native, nlf_root, fp_root, coconet_root, opt_root, staged_mesh_root, metric_mesh_root):
        directory.mkdir(parents=True, exist_ok=True)

    # CARI4D estimate_scale.py explicitly expects a normalized '*_rgba.obj' and emits '*_align.obj'.
    # The provided run_hy3d_recon.py currently emits an '_align.obj' even before metric estimation,
    # so stage an unambiguous normalized name without changing vertices.
    frame_token = "000"
    split_tokens = normalized_mesh.stem.split("_")
    if len(split_tokens) >= 2 and split_tokens[-2].isdigit():
        frame_token = split_tokens[-2]
    staged_dir = staged_mesh_root / f"{sequence}_{frame_token}_rgba"
    staged_normalized = staged_dir / f"{sequence}_{frame_token}_rgba.obj"
    copy_obj_with_materials(normalized_mesh, staged_normalized)

    python = sys.executable
    unidepth_command = [
        python, "prep/unidepth_behave.py", "--wild_video", "--video", str(video), "-o", str(video.parent)
    ]
    if unidepth_model is not None:
        unidepth_command.extend(["--unidepth_model", str(unidepth_model)])
    _run(unidepth_command, cwd=cari4d)
    _run(
        [python, "prep/run_nlf_sepK.py", "-o", str(nlf_root), "--masks_root", str(masks_root),
         "--video", str(video), "--wild_video"],
        cwd=cari4d,
    )
    _run(
        [python, "prep/fit_smplh_global.py", "--wild_video", "--video", str(video),
         "--packed_root", str(packed_root), "--masks_root", str(masks_root), "--nlf_path", str(nlf_root),
         "-o", str(native / "smpl-fit-cache")],
        cwd=cari4d,
    )
    _run(
        [python, "prep/align_monod2hum.py", "--wild_video", "--nlf_path", str(nlf_root) + "-opt",
         "--masks_root", str(masks_root), "--video", str(video),
         "-o", str(native / "align-cache")],
        cwd=cari4d,
    )
    aligned_video = Path(str(video.parent) + "-aligned") / video.name
    if not aligned_video.is_file():
        raise FileNotFoundError(f"CARI4D alignment did not create expected video: {aligned_video}")
    _run(
        [python, "tools/estimate_scale_video.py", "--wild_video", "--video", str(aligned_video),
         "--masks_root", str(masks_root), "--hy3d_root", str(staged_mesh_root), "-o", str(metric_mesh_root)],
        cwd=cari4d,
    )
    metric_dir = metric_mesh_root / staged_normalized.stem
    metric_mesh = metric_dir / staged_normalized.name.replace("_rgba.obj", "_align.obj")
    scale_json = metric_dir / staged_normalized.name.replace("_rgba.obj", "_fp-res-refine.json")
    if not metric_mesh.is_file() or not scale_json.is_file():
        raise FileNotFoundError(
            f"Scale estimation did not emit its source-defined outputs: metric={metric_mesh}, scale={scale_json}"
        )
    _run(
        [python, "prep/fp_hy3d_track.py", "--viz_path", "x", "--wild_video", "--kid", "0",
         "--masks_root", str(masks_root), "--hy3d_root", str(metric_mesh_root), "--video", str(aligned_video),
         "-o", str(fp_root)],
        cwd=cari4d,
    )
    _run(
        [python, "run_horefine.py", "config=learning/configs/cari4d-release.yml",
         "split_file=splits/demo-behave.json", "use_sel_view=True", "render_video=True", "identifier=_demo",
         "use_intermediate=False", "data_name=test-only", f"hy3d_meshes_root={metric_mesh_root}",
         f"masks_root={masks_root}", f"fp_root={fp_root}", f"nlf_root={nlf_root}-opt",
         f"video={aligned_video}", "cam_id=0", "wild_video=True", f"outpath={coconet_root}"],
        cwd=cari4d,
    )
    coconet_result = _unique_result(coconet_root, sequence, "CoCoNet")
    _run(
        [python, "learning/training/opt_refineout.py", f"num_steps={args.num_steps}", "w_acc_v=600",
         "w_contact=300", "save_name=optv2", f"batch_size={args.batch_size}", "opt_rot=True", "opt_trans=True",
         "w_temp=1000", "w_sil=0.002", "w_contact=200.0", "w_pen=2.0", "w_j2d=0.006",
         "opt_smpl_trans=False", "opt_betas=False", "no_wandb=True", f"pth_file={coconet_result}", "wild_video=True",
         "use_input=True", f"video_root={aligned_video.parent}", f"packed_root={packed_root}",
         f"masks_root={masks_root}", f"hy3d_meshes_root={metric_mesh_root}", f"outpath={opt_root}"],
        cwd=cari4d,
    )
    final_result = _unique_result(opt_root, sequence, "joint optimization")
    manifest = {
        "schema": "holosoma.cari4d_native_manifest.v1",
        "video": str(video),
        "aligned_video": str(aligned_video),
        "sequence_name": sequence,
        "gender": subject_gender,
        "cari4d_root": str(cari4d),
        "torch_home": str(torch_home),
        "unidepth_model": str(unidepth_model) if unidepth_model is not None else None,
        "cari4d_result": str(final_result),
        "object_mesh_normalized": str(staged_normalized),
        "object_mesh_metric": str(metric_mesh),
        "scale_json": str(scale_json),
        "upstream_provenance_json": [str(path) for path in upstream_provenance],
    }
    manifest_path = native / "manifest.json"
    write_json(manifest_path, manifest)
    print(f"CARI4D native manifest: {manifest_path}")

    if not args.skip_export:
        export_command = [
            python, str(repository / "tools" / "export_cari4d_sequence.py"), "--native-manifest", str(manifest_path),
            "--output", str(output), "--cari4d-root", str(cari4d), "--sequence-name", sequence,
        ]
        if args.fps is not None:
            export_command.extend(["--fps", str(args.fps)])
        for provenance_path in upstream_provenance:
            export_command.extend(["--upstream-provenance-json", str(provenance_path)])
        _run(export_command, cwd=repository)
        _run(
            [python, str(repository / "tools" / "visualize_object_trajectory.py"), "--sequence", str(output)],
            cwd=repository,
        )


if __name__ == "__main__":
    main()
