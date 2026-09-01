# OMOMO → CARI4D-friendly renderer

This directory is intentionally outside both OMOMO and CARI4D.  It reads the
real OMOMO `sub3_largebox_003` ground-truth motion and produces a static-camera
Blender render plus geometry-derived diagnostic masks.  It never estimates or
changes human/object motion.

All experiment artifacts are written below:

```text
exp/omomo_cari4d/sub03_largebox3/
```

The requested alias `sub03_largebox3` is explicitly mapped to OMOMO's actual
metadata identifier `sub3_largebox_003`; no substitute sequence is used.

## Reproduce

Set paths from the workspace root:

```bash
BLENDER=third_party/blender-3.6.15-linux-x64/blender
ARCHIVE=exp/omomo_cari4d/sub03_largebox3/input/omomo_gt_sequence.npz
ROOT=exp/omomo_cari4d/sub03_largebox3
```

1. Extract GT with OMOMO's official SMPL-H invocation convention:

```bash
PYTHONPATH=third_party/human_body_prior \
third_party/CARI4D/.venv/bin/python \
tools/omomo_cari4d_renderer/prepare_sequence.py
```

2. Search 120 static cameras with 256-pixel two-pass mask renders, then score
all 196 frames at 384 pixels for the winning camera.  The search is resumable:
an interrupted run reloads `camera_candidates.partial.json` and skips completed
azimuth/elevation pairs.

```bash
$BLENDER -b -P tools/omomo_cari4d_renderer/blender_pipeline.py -- \
  search --sequence-archive "$ARCHIVE" --output "$ROOT/camera_search"
```

3. Render the complete 1280×1280 sequence and diagnostic passes:

```bash
$BLENDER -b -P tools/omomo_cari4d_renderer/blender_pipeline.py -- \
  render --sequence-archive "$ARCHIVE" \
  --camera-config "$ROOT/camera_search/best_camera.json" \
  --resolution 1280 --output "$ROOT/cari4d_friendly"
```

4. Encode videos, make the comparison/contact sheet, copy Top-K frames, and
pack Blender GT masks in official CARI4D HDF5 layout:

```bash
third_party/CARI4D/.venv/bin/python \
tools/omomo_cari4d_renderer/postprocess_render.py
```

The camera search score is:

```text
1.0*object_size + 0.5*object_visibility + 1.5*view_informativeness
+ 0.5*human_visibility - 2.0*object_occlusion - 2.0*crop
```

Object occlusion is measured by rendering the object alone (Pass A) and the
human with the object (Pass B), then computing
`1 - visible_object_area / object_only_area`.

Camera selection also enforces that at least one sampled reconstruction frame
has a 1280-pixel object bbox whose width *and* height are at least 200 pixels,
object occlusion below 10%, human visibility at least 98%, object visibility at
least 90%, and view informativeness at least 0.65.  Among passing cameras, the
whole-trajectory camera score is the tie-break.  This prevents an excellent
average view from winning while leaving the object too small for Hunyuan.

The full-resolution human/object PNG masks are explicit unlit ID renders under
the same fixed camera and depth test.  This avoids a Blender 3.6 surfaceless
EEVEE issue in which legacy compositor IDMask-to-PNG nodes may not write files.
Depth and normals remain untouched diagnostic compositor passes; neither path
changes the RGB image.

## Model-resource note

OMOMO stores 16 betas, while the locally licensed SMPL-H resource contains 10
shape directions.  The first 10 betas are evaluated; the final 6 cannot affect
that model.  Every pose, human translation, object vertex, object scale,
rotation and translation remains exact OMOMO GT.  This limitation is recorded
in the generated input metadata rather than hidden.
