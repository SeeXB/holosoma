# Human video → CARI4D → canonical HOI → IsaacLab USD

This pipeline deliberately has two Python environments. Stage A runs CARI4D and writes files; Stage B runs
Isaac Sim/Isaac Lab and consumes only the canonical OBJ. It does not install packages, change CUDA/PyTorch, or
modify either third-party project.

This stage stops at human/object reconstruction and object USD conversion. It does **not** perform G1/GMR
retargeting, RL tracking, or policy training.

## Source audit: the actual CARI4D serialization

The adapter follows checked-out CARI4D commit `71fa7cbe46081467edadd11ab534b0c14aa9d913`, rather than README
field-name guesses:

- `third_party/CARI4D/run_horefine.py:562-569` saves a native `.pth` with top-level keys `gt`, `pr`, and `in`.
  Its preliminary `pr` fields are `pose_abs`, `smpl_pose`, `smpl_t`, `frames`, `betas`, `verts`, and
  `contact_logits`.
- `third_party/CARI4D/learning/training/opt_refineout.py:496-522` overwrites the final `pr` with
  `pose_abs [T,4,4]`, `smpl_pose [T,72]`, `smpl_t [T,3]`, `betas [T,10]`, and `frames`, then writes
  `<outpath>/<experiment>/<sequence>.pth`. It also pickles a `TrainState` instance; the exporter temporarily
  exposes the explicitly selected `--cari4d-root` while loading so Python can resolve that source class.
- `third_party/CARI4D/lib_smpl/__init__.py:21-51` defines the official 72D→156D SMPL-H mapping. The exporter
  loads the checked-out `GRAB_MEAN_HAND` prior, outputs `poses [T,156]`, and also preserves the unmodified source
  array as `poses_cari4d_raw`.
- `third_party/CARI4D/tools/estimate_scale.py:61-129` says its source mesh is normalized, serializes scalar
  `best_scale`, and writes `_align.obj` with `vertices * best_scale`. Therefore that metric OBJ must not be
  scaled again.
- `third_party/CARI4D/run_horefine.py:120-129` and
  `third_party/CARI4D/learning/training/opt_refineout.py:179-227` subtract the arithmetic vertex mean before
  applying `pose_abs`. `object_metric.obj` repeats exactly that canonicalization so the saved transforms remain
  valid without translation compensation.
- `third_party/CARI4D/Utils.py:530-548` constructs `(x,y,z)` from image `(u,v,depth)` in OpenCV camera space.
  For a custom monocular video there is no independently estimated static world frame. The canonical metadata
  therefore truthfully says `cari4d_metric_camera` (`+X` right, `+Y` down, `+Z` forward), not a guessed Isaac
  world frame.

No CARI4D→Isaac axis rotation is applied in this stage. Both human and object remain in the same CARI4D metric
camera frame. `T_isaac_from_cari4d` is stored as `null`; a later consumer must apply one shared, explicit
transform to both trajectories. Metadata carries both the machine-readable `coordinate_frame` and the explicit
`coordinate_system` axis description.

## Stage A0: generic RGB-video preparation

The primary preprocessing entry point is generic and defaults to CARI4D's official
SAM3 text-prompted segmentation. It stages any input filename into the native
`<sequence>.0.color.mp4` convention and validates that every frame has a non-empty
human and object mask:

```bash
bash scripts/prepare_video_for_cari4d.sh \
  --video /path/to/input.mp4 \
  --output-root outputs/my_sequence/preprocess \
  --gender male \
  --object-name cardboardbox \
  --human-prompt person \
  --object-prompt "cardboard box" \
  --sam3-python /path/to/sam3/environment/bin/python
```

If the active shell Python does not provide OpenCV/HDF5, set
`PREPROCESS_PYTHON=/path/to/cari4d/python` for the lightweight staging and
validation process. This is independent from `--sam3-python`.

SAM3 intentionally runs in its own environment. The command calls CARI4D's
checked-out `prep/run_sam3_masks.py`; it does not install or copy SAM3 code.
`--mask-backend existing` accepts already packed masks. An explicit
`--mask-backend color-key` exists only for synthetic/green-screen inputs with
known hue ranges. It is never auto-detected, and provenance records
`sample_specific_backend: true` so a synthetic sample adapter cannot be confused
with the general RGB-video method.

The generated sequence alias uses a CARI4D-compatible subject token solely
because the current upstream code indexes that token to select male/female
SMPL-H. The report retains the unrestricted source-video path and the explicit
gender/object name.

Object reconstruction and independent 2D pose preprocessing remain the official
CARI4D stages described in `third_party/CARI4D/docs/custom_video.md`:

```bash
# Hunyuan3D environment
python third_party/CARI4D/prep/run_hy3d_recon.py --video <staged-video> \
  --masks_root <masks-root> --hy3d_root <mesh-root> --blender_path <blender>

# CARI4D/Sapiens environment
python third_party/CARI4D/prep/run_sapiens_pose.py --video <staged-video> \
  --masks_root <masks-root> --packed_root <packed-root>
```

## Stage A: reconstruction and canonical export

Activate the CARI4D environment first. The wrapper executes the same seven commands as CARI4D's
`scripts/demo-custom.sh`, but parameterizes all output roots and discovers the actual result directory after it
is written.

```bash
conda activate cari4d

bash scripts/run_video_to_hoi.sh \
  --video /data/Date03_Sub01_suitcase_custom.0.color.mp4 \
  --output outputs/suitcase_custom \
  --masks-root /data/cari4d/masks \
  --packed-root /data/cari4d/packed \
  --object-mesh-normalized /data/cari4d/meshes/normalized_suitcase.obj \
  --upstream-provenance-json outputs/my_sequence/preprocess/validation/video_preparation_report.json
```

The filename must follow CARI4D's native `<sequence>.0.color.mp4` convention, and the sequence currently needs
at least three underscore-separated tokens because upstream code indexes subject and object tokens. The subject
token must also exist in CARI4D's `_sub_gender` table (for example `Sub01=male`, `Sub06=female`); the wrapper
refuses to guess gender. The checked-out README places the licensed models under `data/smpl/smplh/`, but
`lib_smpl/const.py` and `lib_smpl/smpl_module.py` actually open `data/smpl/SMPLH_<gender>.pkl` directly. Keep the
licensed files in the README directory and expose symlinks at the source-read path; preflight checks the latter
so this mismatch cannot fail halfway through reconstruction. Joint optimization has a second convention:
`smplx.create(model_path="data/smpl", model_type="smplh", gender=...)` resolves an uppercase filename under
`data/smpl/smplh/SMPLH_<GENDER>.pkl`. Preflight checks that path too; a symlink to the same licensed lowercase
README file avoids duplicating the 130 MB model only when that pickle already contains the four fields required
by `smplx`: `hands_componentsl`, `hands_componentsr`, `hands_meanl`, and `hands_meanr`. Some older MANO-site
SMPL-H archives omit those fields and distribute the matching hand bases in `MANO_LEFT.pkl`/`MANO_RIGHT.pkl`;
in that case, keep the original files and create a separate compatibility copy containing those exact licensed
hand fields. Preflight inspects the copy rather than accepting a path-only check. If masks,
keypoints, a normalized mesh, model weights, SMPL-H models, UniDepth, or VolumetricSMPL are missing, the wrapper
stops in preflight and prints the relevant official preprocessing command. It does not invoke pip.

CoCoNet additionally constructs `dinov2_vitb14` and `dinov2_vits14` through `torch.hub` (the model names come
from `learning/training/training_config.py`, not the release YAML). The wrapper deliberately uses
`third_party/CARI4D/.cache/torch` by default and checks both the cached DINOv2 source and pretrained checkpoints
before starting reconstruction. This keeps dynamic downloads out of the user's global home cache. Override the
location with `--torch-home` only when an already-populated cache should be reused. Joint optimization is run
with CARI4D's `no_wandb=True`, because inference output must not depend on an interactive WandB login.

Joint optimization also calls `VolumetricSMPL.attach_volume()`, whose checked-out implementation dynamically
loads `VolumetricSMPL_smpl_<gender>.ckpt` from the upstream `dev/models` branch. Merely cloning the Python code
as instructed by the CARI4D README does not populate that checkpoint. The same preflight therefore requires it
under `<TORCH_HOME>/hub/checkpoints` before the long run starts.

CARI4D's current `run_hy3d_recon.py` names its normalized output `_align.obj`, while `estimate_scale.py` only
handles `_rgba.obj` correctly. The wrapper stages a vertex-identical `_rgba.obj` under the sequence output,
records that exact path in `cari4d_native/manifest.json`, and leaves CARI4D's algorithms untouched.

The checked-out `align_monod2hum.py` also inherits an author-machine default `-o /home/xianghuix/...`; its
multiprocessing parent does not propagate a child permission failure. The wrapper therefore passes a sequence-local
`cari4d_native/align-cache` explicitly and additionally requires the expected aligned color/depth files before it
continues.

To export an already-computed result, pass all provenance-bearing artifacts explicitly:

```bash
python tools/export_cari4d_sequence.py \
  --cari4d-result /path/to/final_opt/sequence.pth \
  --object-mesh-normalized /path/to/normalized_rgba.obj \
  --object-mesh-metric /path/to/metric_align.obj \
  --scale-json /path/to/object_fp-res-refine.json \
  --video /path/to/sequence.0.color.mp4 \
  --output outputs/sequence
```

The final `.pth` does not serialize mesh or scale paths. Consequently the exporter refuses to infer those paths
from names. A wrapper-created manifest can replace the four artifact arguments:

```bash
python tools/export_cari4d_sequence.py \
  --native-manifest outputs/sequence/cari4d_native/manifest.json \
  --output outputs/sequence
```

CARI4D also does not serialize FPS. Supply `--video` (preferred) or an explicit `--fps`; there is no hidden
30 FPS fallback.

When an external object asset is already known to be in meters (for example a simulation asset supplied with
the source dataset), do not send it through UniDepth scale estimation. Stage a textured, vertex-identical
tracking template and make the bypass explicit during export:

```bash
python tools/prepare_known_object_mesh_for_cari4d.py \
  --input /path/to/known_metric_object.obj \
  --output-root outputs/sequence/cari4d_native/metric_mesh \
  --sequence-name <sequence> \
  --input-unit meter

# Run FoundationPose, CoCoNet, and joint optimization against the emitted _align.obj.

python tools/export_cari4d_sequence.py \
  --cari4d-result /path/to/final_opt/sequence.pth \
  --object-mesh-metric outputs/sequence/cari4d_native/metric_mesh/<sequence>_000_rgba/<sequence>_000_align.obj \
  --object-mesh-is-metric \
  --video /path/to/sequence.0.color.mp4 \
  --output outputs/sequence
```

The preparation utility only adds a constant texture when the asset lacks one; it does not move or scale any
vertex. The exporter records `known_metric_asset_no_scale_estimation`, applies scale `1.0`, and centers the
metric mesh in the same arithmetic-mean canonical frame used by CARI4D. This mode is mutually exclusive with
`--scale-json` and `--derive-scale-from-metric-mesh`.

NVIDIA's official demo archive is a special case: it distributes an exactly corresponding normalized OBJ and
metric OBJ but omits the scale JSON. For that archive only, recover the unique scalar with an explicit opt-in:

```bash
python tools/export_cari4d_sequence.py \
  --cari4d-result /path/to/final_opt/sequence.pth \
  --object-mesh-normalized /path/to/official_normalized.obj \
  --object-mesh-metric /path/to/official_metric.obj \
  --derive-scale-from-metric-mesh \
  --video /path/to/sequence.0.color.mp4 \
  --output outputs/sequence
```

This mode fits one scalar over all corresponding vertex coordinates and still rejects the pair unless
`V_metric ≈ s * V_normalized` within the same strict tolerance. It is never selected implicitly, and metadata
records this provenance instead of claiming that a native JSON existed.

The scale check prints normalized extent, `best_scale`, raw metric extent, and canonical metric extent. If
`metric_extent != normalized_extent * best_scale`, export aborts instead of risking double scaling. The metric
source mesh is centered but never rescaled a second time.

Some meshes in NVIDIA's official demo contain invalid OBJ face-normal index `0` values, which Isaac's importer
rejects. Canonical export detects these explicitly and removes only the invalid normal references on affected
faces (vertices, UVs, and topology are unchanged); the repair count is printed and stored as
`obj_invalid_zero_normal_face_indices_repaired`. `object_original.obj` is not sanitized.

Validate and visualize without Isaac Lab:

```bash
python tools/validate_cari4d_export.py --sequence outputs/sequence
python tools/visualize_object_trajectory.py --sequence outputs/sequence
```

The validator reads `source_video_num_frames` from `meta/sequence.json` when available, so an independent rerun
retains the human/object/video alignment check instead of replacing it with an unknown count.

The visualization writes `validation/object_trajectory_keyframes.png` for the first, middle, and last object
poses.

## Stage B: IsaacLab MeshConverter

Activate the existing Isaac environment. On this workstation the verified launch path is:

```bash
source scripts/source_isaacsim_setup.sh

python tools/convert_object_to_usd.py \
  --input outputs/sequence/object/object_metric.obj \
  --output outputs/sequence/object/object.usd \
  --collision-approximation convexDecomposition \
  --mass 1.0 \
  --headless
```

The script uses the locally installed API:

```python
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
```

It always passes mesh scale `(1,1,1)`. Collision choices are `convexDecomposition` (default), `convexHull`, and
`none`. A rigid body and mass are authored even when the default mass is used; metadata records
`mass_is_placeholder: true`. Use `--mass-source measured` only when the value came from an external measurement.

Re-run validation independently in the same Isaac environment:

```bash
python tools/validate_object_usd.py \
  --usd outputs/sequence/object/object.usd \
  --source-obj outputs/sequence/object/object_metric.obj \
  --expected-collision convexDecomposition \
  --expected-mass 1.0 \
  --headless
```

Validation checks the default prim, visual Mesh prim, enabled CollisionAPI and approximation, RigidBodyAPI,
MassAPI, authored mass, and per-axis OBJ/USD bounding boxes. A >1% bbox difference fails by default, and obvious
100×/1000× discrepancies always fail.

## Output schema

```text
outputs/<sequence>/
├── human/
│   ├── smplh.npz
│   └── metadata.json
├── object/
│   ├── object_original.obj
│   ├── object_metric.obj
│   ├── initial_pose.npy
│   ├── trajectory.npz
│   ├── metadata.json
│   └── object.usd
├── meta/
│   └── sequence.json
└── validation/
    ├── export_report.json
    ├── object_trajectory_keyframes.png
    └── usd_report.json
```

`human/smplh.npz` contains `poses`, `trans`, `betas`, `fps`, `frame_ids`, plus the unmodified
`poses_cari4d_raw`. `object/trajectory.npz` contains `translation`, `rotation_matrix`, `quaternion_wxyz`
(scalar-first), `transform`, `fps`, and `frame_ids`. `initial_pose.npy` is exactly `transform[0]`.

Isaac kinematic replay is intentionally not included in this first version: it would require choosing a
CARI4D-camera→Isaac-world transform, and this pipeline refuses to guess one. The dependency-free three-keyframe
visualization is the implemented geometry/trajectory check.

## Verified samples on this workstation

### Target large-box video

The requested target video
`/mnt/sdadrive/shixiongbo/omomo/motion_videos/sub3/sub3_largebox_003.mp4` was processed by the generic
RGB-video path (SAM2 box-prompt masks, Sapiens pose, Hunyuan3D object reconstruction, UniDepth,
FoundationPose, CoCoNet, and the full 3000-step CARI4D optimization). No OMOMO motion, object pose, or object
mesh label was used as a reconstruction input. The OMOMO reference was loaded only after export by the
diagnostic comparison script and is marked `evaluation_only_not_reconstruction_input` in its report.

The native optimized result has top-level keys `gt`, `pr`, and `in`. Its actual final prediction fields are:

```text
pr.pose_abs:       [196, 4, 4]
pr.smpl_pose:      [196, 72]
pr.smpl_t:         [196, 3]
pr.betas:          [196, 10]
pr.frames:         196 entries
pr.contact_logits: [196, 2]
```

Canonical and USD validation both pass:

```text
video / human / object frames: 196 / 196 / 196 (IDs 0...195)
fps:                           30
SMPL-H poses:                  [196, 156]
raw CARI4D poses preserved:    [196, 72]
object transforms:             [196, 4, 4]
CARI4D metric scale:           0.6444827586
normalized OBJ extent:         [0.863747, 0.811591, 1.995064]
metric OBJ extent:             [0.556671, 0.523056, 1.285784] m
USD extent:                    [0.556670994, 0.523055986, 1.285784006] m
max rotation orthogonality:    4.386e-7
max quaternion norm error:     5.960e-8
collision:                     convexDecomposition
mass:                          1.0 kg (placeholder)
canonical / USD validation:    PASS / PASS
```

Passing schema and USD validation does not imply that this monocular reconstruction is accurate enough for
RL. An evaluation-only comparison against the original OMOMO trajectory found:

```text
root-centered body-22 MPJPE after one global rotation: 0.0532 m
object position RMSE after rigid alignment:             0.2452 m
object path length, CARI4D / reference:                 6.936 / 3.679 m
object orientation profile correlation:                -0.048
object-pelvis distance RMSE:                            0.1966 m
generated / reference object mesh volume ratio:        approximately 7.31x
```

The human body articulation is usable as a first-pass reconstruction, but the recovered box is elongated and
the object 6DoF trajectory is substantially noisier than the reference. Therefore this particular result must
not yet replace the original trajectory for policy training. Also, CARI4D's final prediction stores a 72D
body pose; the canonical exporter preserves it as `poses_cari4d_raw` and expands to SMPL-H 156D with CARI4D's
GRAB mean-hand prior. The exported hand pose is consequently not per-frame hand articulation recovered from
the video.

Relevant artifacts are under `outputs/human_video_sub3_largebox_003`, including
`validation/export_report.json`, `validation/usd_report.json`,
`validation/cari4d_vs_intermimic_report.json`, and
`validation/cari4d_vs_intermimic_comparison.png`.

The object stage was subsequently rerun with OMOMO's known meter-scale `largebox.obj`, while keeping all human
and object motion labels excluded from reconstruction. The asset was used only as canonical shape; FoundationPose,
CoCoNet, and the full joint optimization still estimated every 6DoF pose from the video. This result is under
`outputs/human_video_sub3_largebox_003_known_obj`:

```text
video / human / object frames:                    196 / 196 / 196
known mesh scale operation:                       none (scale 1.0)
OBJ / USD AABB:                                   [0.471154, 0.458730, 0.407895] m
OBJ/USD bbox ratio:                               [1.000000006, 1.000000003, 0.999999989]
collision / mass:                                 convexDecomposition / 1.0 kg placeholder
canonical / USD validation:                       PASS / PASS
root-centered body-22 MPJPE:                      0.0558 m
object position mean / RMSE after rigid alignment: 0.1565 / 0.1796 m
object path length, CARI4D / reference:           5.706 / 3.679 m
object speed-profile correlation:                 0.632
object relative-rotation profile correlation:     0.983
object-pelvis distance MAE / correlation:         0.0441 m / 0.979
```

The known shape fixes the severe geometry and rotation-profile failure and makes the human-object relative
distance closely match the reference. The remaining approximately 16 cm aligned object-center error and excess
path length show that the monocular trajectory is still noisier than the original motion; the report should be
consulted before using it as an RL reference rather than treating schema validation as an accuracy guarantee.

### Multi-frame collision-box diagnostic

`tools/estimate_multiframe_collision_box.py` tests whether CARI4D's metric depth, masks, and object poses are
sufficient to produce collision geometry. It motion-compensates depth into the pose source's canonical object
frame, rejects one-frame voxel outliers, exports a robust OBB and convex hull, and independently computes an
occlusion-aware silhouette visual hull. Human-mask pixels are treated as unknown during carving. The pose format
is always explicit; a mesh passed with `--evaluation-mesh` is loaded only after estimation and cannot affect the
result.

Example using the original Hunyuan-based FoundationPose trajectory:

```bash
PYTHONPATH=third_party/CARI4D third_party/CARI4D/.venv/bin/python \
  tools/estimate_multiframe_collision_box.py \
  --video-prefix outputs/human_video_sub3_largebox_003/preprocess/videos-aligned/Custom_Sub01_box_sub3largebox003 \
  --masks-h5 outputs/human_video_sub3_largebox_003/preprocess/masks/Custom_Sub01_box_sub3largebox003_masks_k0.h5 \
  --pose-file outputs/human_video_sub3_largebox_003/cari4d_native/fp-hy3d-track/Custom_Sub01_box_sub3largebox003_all.pkl \
  --pose-format foundationpose-pkl \
  --output outputs/human_video_sub3_largebox_003/validation/multiframe_collision/old_hunyuan_fp
```

For this fixed-camera large-box clip, the typical per-frame visible span is close to the real box size, but the
enclosing multi-frame estimates are not reliable:

```text
reference mesh OBB:                    [0.3351, 0.3377, 0.3601] m
old Hunyuan pose, median visible span: [0.3189, 0.3549, 0.3618] m
old Hunyuan pose, depth-fusion OBB:    [0.3582, 0.4973, 0.5779] m
old joint-opt pose, depth-fusion OBB:  [0.3938, 0.4869, 0.5399] m
known-mesh FP pose, depth-fusion OBB:  [0.3373, 0.3956, 0.6408] m
known-mesh FP pose, visual-hull OBB:   [0.3184, 0.4434, 0.9311] m
```

All three runs are marked `UNSTABLE`. The upper-bound known-mesh pose still produces a 64 cm fused long axis,
so the failure is not explained only by the erroneous Hunyuan mesh. Per-frame monocular depth/pose disagreement
smears motion-compensated surfaces, while the object does not rotate enough relative to the single fixed camera
to constrain its hidden depth with a visual hull. The median visible span is useful as a size hint for a known
cuboid prior, but it is not an enclosing collision volume and must not be sent directly to MeshConverter. Reports,
point clouds, OBJ proxies, and plots are under
`outputs/human_video_sub3_largebox_003/validation/multiframe_collision`.

### Official CARI4D sample

The full official wild-video sample `Date03_Sub01_gas_wild002` was run through UniDepth alignment,
FoundationPose tracking, CoCoNet inference, the complete 3000-step joint optimization, canonical export, and
the local IsaacLab MeshConverter. Its final output is `outputs/cari4d_official_wild_full`:

```text
video / human / object frames: 1209 / 1209 / 1209 (IDs 0...1208)
fps:                           25.0209999
SMPL-H poses:                  [1209, 156]
raw CARI4D poses preserved:    [1209, 72]
object transforms:             [1209, 4, 4]
derived official mesh scale:   0.3222222354
metric OBJ extent:             [0.334091, 0.642821, 0.331984] m
USD extent:                    [0.334090993, 0.642820984, 0.331983998] m
max rotation orthogonality:    7.754e-7
max quaternion norm error:     5.960e-8
collision:                     convexDecomposition
mass:                          1.0 kg (placeholder)
canonical / USD validation:    PASS / PASS
```

The earlier Stage-B-only geometry smoke test remains at `outputs/cari4d_official_geometry_smoke`:

`outputs/cari4d_official_geometry_smoke` was generated from NVIDIA's official `cari4d-demo.zip` gas-cylinder
normalized/metric mesh pair (archive SHA-256 is recorded in object metadata), then converted with the local
IsaacLab MeshConverter. Results:

```text
vertices:              24463
faces:                 48865
derived demo scale:    0.3222222354
normalized extent:     [1.036834, 1.994960, 1.030295]
metric OBJ extent:     [0.334091, 0.642821, 0.331984] m
USD extent:            [0.334090993, 0.642820984, 0.331983998] m
collision:             convexDecomposition
mass:                  1.0 kg (placeholder)
USD validation:        PASS
```

The archive itself does not include a final CARI4D inference `.pth` or its scale JSON, so that directory remains
explicitly marked as a geometry/Stage-B smoke test. The full directory above comes from the actual inference and
joint-optimization run, not fabricated trajectory tensors.
