# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplh, mocap, lafan) and task types including robot-only motion, object interaction, and climbing.

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

## Semantic Keyframes Before Retargeting

The optional semantic-keyframe stage preserves the input boundary of the human-video pipeline:

```text
human video -> video-derived SMPL-X human/object trajectory -> semantic keyframes -> retargeting
```

The stage does not read a retargeted or dataset-specific `.pt` file. It takes the SMPL-X `.npz` produced from the human video, asks a vision-language model (VLM) to identify the ordered semantics of the box-carry sequence, then resolves the declarative signal rules to deterministic frame indices from that same SMPL-X trajectory. The VLM never emits frame indices or executable code.

Set an OpenAI-compatible vision endpoint in `.env`:

```bash
OPENAI_BASE_URL=https://your-endpoint.example/v1
OPENAI_API_KEY=your-key
OPENAI_MODEL=your-vision-model
```

Run the stage before retargeting:

```bash
python -m holosoma_retargeting.semantic_keyframes \
    --video /path/to/sub3_largebox_003.mp4 \
    --smplx-file /path/to/sub3_largebox_003.npz \
    --model-dir /path/to/body_models \
    --output semantic_keyframes/sub3_largebox_003.json
```

An installed package also provides the `holosoma-semantic-keyframes` command. The output schema remains compatible with the previous GMR semantic-keyframe JSON: each event contains a deterministic `trigger_frame` and window. Invalid VLM output is repaired through an exact field allowlist: already-valid fields and events are immutable, and any patch that includes an unrequested field is rejected locally. For auditability, the command saves every raw VLM response, the accepted declarative event plan, and the accepted repair paths next to the output file.

The repository includes the migrated `sub3_largebox_003` example output under `holosoma_retargeting/demo_data/semantic_keyframes/`, including both the resolved keyframes and the accepted VLM event plan.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```

**Note**: Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Semantic-Keyframe-Aware OmniRetarget

The supported semantic main line changes only the existing Laplacian residual weights and, optionally, the iteration cap at exact semantic triggers. It does not add a second objective or solver stage, lower ordinary-frame compute, or change topology, samples, and physical constraints:

| Mode | SQP budget | Semantic residual weights |
| --- | --- | --- |
| `original` | official 50/10 | off |
| `uniform` | frame 0 = 50, all other frames = N | off |
| `uniform2_semantic_weight_uniform` | 50/2 | registered equal-strength Legacy weights |
| `uniform2_original_objective_semantic_budget` | 50/2; all nonzero exact triggers 4 | off |
| `uniform2_semantic_weight_semantic_budget` | 50/2; all nonzero exact triggers 4 | registered Legacy weights |
| `uniform2_semantic_weight_random_budget` | 50/2; K matched random ordinary frames 4 | registered Legacy weights |
| `uniform2_semantic_weight_full_event` | 50/2 | all projected events and JSON body parts |
| `uniform2_semantic_weight_full_event_approach_body_only` | 50/2 | full-event weights, but approach weights pelvis only and never object neighbors |
| `uniform2_semantic_weight_full_event_transition_truncated` | 50/2 | each event owns `[trigger, min(end, next trigger - 1)]` |
| `uniform2_semantic_weight_full_event_transition_truncated_budget` | 50/2; nonzero exact triggers N | transition-truncated full-event weights |
| `uniform2_semantic_weight_full_event_budget` | 50/2; nonzero exact triggers N | all projected events and JSON body parts |
| `uniform2_semantic_weight_full_event_random_budget` | 50/2; K matched random ordinary frames N | all projected events and JSON body parts |

For the full-event refinement modes, `N` is restricted to the registered set
`{2, 4, 6, 8, 10}`. The historical `uniform2_semantic_weight_uniform` mode
continues to use only contact/lift/place/release and remains trajectory-level
backward compatible.

The active optimizer reads a deterministic projection of `semantic_v2.json`: event name, window, trigger frame, body parts, trigger/end rules, and rationale are retained. Historical criticality fields are ignored and never affect weighting or scheduling. The JSON is not regenerated.

From `src/holosoma_retargeting/holosoma_retargeting`, a Semantic Budget run is:

```bash
python examples/robot_retarget.py \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data-format smplh \
    --data-path demo_data/OMOMO_new \
    --save-dir benchmark_results_semantic_budget/runs/semantic_budget \
    --semantic.mode uniform2_semantic_weight_semantic_budget \
    --semantic.semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --semantic.profile-dir benchmark_results_semantic_budget/runs/semantic_budget
```

`original` and `uniform` never open the semantic JSON. For example, the strict compatibility baseline is:

```bash
python examples/robot_retarget.py \
    --task-type object_interaction \
    --task-name sub3_largebox_003 \
    --data-format smplh \
    --data-path demo_data/OMOMO_new \
    --save-dir benchmark_results/original \
    --semantic.mode original \
    --semantic.profile-dir benchmark_results/original
```

Run the registered Legacy compatibility check, Semantic/Random/Original budget controls, unified evaluations, and plots with:

```bash
python examples/benchmark_semantic_budget.py \
    --task-name sub3_largebox_003 \
    --data-path demo_data/OMOMO_new \
    --semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --output-dir benchmark_results_semantic_budget
```

Random seeds 0–4 use the same K extra slots and exclude frame 0 plus every true trigger ±3. Use `--force` to rerun known artifacts or `--aggregate-only` to rebuild the tables and plots. Historical benchmark directories are retained as read-only experiment records.

Run the registered Full-Event refinement curve with:

```bash
python examples/benchmark_full_event_semantic_budget.py \
    --task-name sub3_largebox_003 \
    --data-path demo_data/OMOMO_new \
    --semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --output-dir benchmark_results_full_event_semantic_budget
```

Every actual solve at a nonzero semantic trigger is logged with unweighted
Part, Local, Edge, and global Laplacian errors. The registered frame-34
physical diagnostic additionally records the unchanged MuJoCo 0.1 m distance
query, shoulder-box signed distance, illegal penetration, and q-update norm at
each accepted iterate. Intermediate physical failures do not stop the fixed
2/4/6/8/10 curve, and no fallback trajectory is substituted. Trace-evaluation
overhead is excluded from the reported retargeting wall time.

Run the causal approach body-only validation with:

```bash
python examples/benchmark_full_event_approach_body_only.py \
    --task-name sub3_largebox_003 \
    --data-path demo_data/OMOMO_new \
    --semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --output-dir benchmark_results_full_event_approach_body_only
```

On the registered sequence, the realized interaction-mesh topology has zero
direct pelvis-to-object edges at every active approach frame. Consequently,
this body-only intervention is exactly trajectory-equivalent to FullEvent-B2;
the diagnostic is retained to make that null result and its topology cause
reproducible.

Run the generic transition-truncation diagnostic with:

```bash
python examples/benchmark_full_event_transition_truncation.py \
    --task-name sub3_largebox_003 \
    --data-path demo_data/OMOMO_new \
    --semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --output-dir benchmark_results_full_event_transition_truncation
```

This policy uses the same rule for every event: spatial weighting starts at
the event trigger and ends at the earlier of its window end or the frame
before the next event trigger. On the registered sequence it removes the
frame-34 penetration while retaining the fixed 440-solve budget; the complete
semantic-quality tradeoff is recorded in the benchmark report.

Run the B2/B4/B6/B8/B10 exact-trigger budget curve with:

```bash
python examples/benchmark_transition_truncation_budget_curve.py \
    --task-name sub3_largebox_003 \
    --data-path demo_data/OMOMO_new \
    --semantic-keyframe-path demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
    --output-dir benchmark_results_full_event_transition_truncation
```

The registered final recipe is
`uniform2_semantic_weight_full_event_transition_truncated_budget` with
`exact_trigger_budget=4`: frame 0 uses 50 iterations, the seven nonzero
semantic triggers use at most 4, and every ordinary frame uses 2. B4 is the
lowest-compute point within 0.1% of the best feasible All-event Part error on
the registered curve and preserves zero official penetration.

Hand/object distance is measured against the existing fixed object surface samples, so it is a reproducible surface approximation rather than exact triangle-mesh distance. The official optimizer has no velocity-limit constraint; velocity violation is therefore reported as unavailable unless `--semantic.rescue-velocity-limit-per-frame` is explicitly configured. Self-collision is likewise reported as unavailable unless collision pairs are configured. No threshold is tuned from benchmark results.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Data Preparation

We provide `demo_data/` for fast testing. To test on more motion sequences, please follow the instructions below to download and prepare the data.

### OMOMO

Our pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `demo_data/OMOMO_new`

The data should contain `.pt` files.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `demo_data/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan --output_dir ../demo_data/lafan
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. The SMPL-X models structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
# Install dependencies
cd holosoma_retargeting/data_utils/
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
cd ../

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

This will convert the AMASS `.npz` files to `.npz` format with global joint positions and height information.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path demo_data/amass_smplx_processed --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/amass_smplx_processed --task-type robot_only --data_format smplx --save_dir demo_results_parallel/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

```bash
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/amass_smplx/HumanEva_S1_Box_1_stageii_original.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above (Single Sequence Motion Retargeting or Batch Processing for Motion Retargeting).

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training. The conversion script takes the retargeted `.npz` files as input and outputs converted files with the specified frame rate and format.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)
