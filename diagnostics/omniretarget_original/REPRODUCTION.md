# Original OmniRetarget reproduction

- Baseline source: `fb835ec8cb6ee48f483ce567586625e5fae1ae1f`
  (`origin/main` before the semantic-keyframe commits).
- Input: `demo_data/OMOMO_new/sub3_largebox_003.pt`.
- Input SHA256: `dead7293077ca1e1ea1a344cf416182463fa2f4d66f4e304cc846866913a4844`.
- Task: `object_interaction`, format `smplh`, robot `g1`.
- Frames: 196 at 30 Hz before conversion; 325 at 50 Hz after conversion.
- No semantic configuration or semantic-keyframe JSON exists in the exported
  baseline source.

## Retarget command

Run from the exported baseline's
`src/holosoma_retargeting/holosoma_retargeting` directory:

```bash
conda run -n hsretargeting python examples/robot_retarget.py \
  --data_path demo_data/OMOMO_new \
  --task-type object_interaction \
  --task-name sub3_largebox_003 \
  --data_format smplh \
  --save-dir /home/zongyouyu/sxb/holosoma/diagnostics/omniretarget_original/retarget_raw
```

Raw output SHA256:
`22ca869d0996cb0ebb15c9c9abef446cccbbe894e70f85239cb172f034d22117`.

## RL conversion

The pre-semantic converter has an upstream metadata bug: it interprets the
retargeter's `fps=30` as a frame duration and evaluates `round(1 / 30)`, which
becomes zero. The current converter contains a compatibility-only correction
that accepts both historical frame-duration metadata and proper FPS metadata.
That corrected converter was used without changing the retargeted trajectory.

```bash
conda run -n hsretargeting python data_conversion/convert_data_format_mj.py \
  --input_file /home/zongyouyu/sxb/holosoma/diagnostics/omniretarget_original/retarget_raw/sub3_largebox_003_original.npz \
  --output_fps 50 \
  --output_name /home/zongyouyu/sxb/holosoma/diagnostics/omniretarget_original/rl/sub3_largebox_003_original_mj_fps50_w_obj.npz \
  --data_format smplh \
  --object_name largebox \
  --has_dynamic_object \
  --headless \
  --once
```

RL output SHA256:
`e096487ecfb3a52d1039651d9e5bc0ee248a8caf2d34570d9adb792b1533ea46`.

The converted file has all 13 expected Holosoma arrays, no NaN values, and
matches the repository-registered object trajectory to numerical precision
(`object_pos_w` maximum absolute difference `2.98e-7 m`). Relative to the
registered robot trajectory, maximum body-position difference is `2.58 mm` and
body-position RMSE is `0.100 mm`, consistent with a fresh SQP reproduction.

## Original WBT baseline

The training run uses the repository's original object-WBT experiment and
overrides only the motion path and the descriptive training name:

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object \
  logger:wandb \
  --training.name=g1_29dof_wbt_omniretarget_original \
  --training.num-envs=4096 \
  --algo.config.num-learning-iterations=30000 \
  --command.setup_terms.motion_command.params.motion_config.motion_file=/home/zongyouyu/sxb/holosoma/diagnostics/omniretarget_original/rl/sub3_largebox_003_original_mj_fps50_w_obj.npz
```

- tmux session: `omniretarget_original_wbt_30k_wandb`
- W&B run: <https://wandb.ai/yumou0319-/WholeBodyTracking/runs/yofzfd5c>
- Local config/run directory:
  `logs/WholeBodyTracking/20260820_153924-g1_29dof_wbt_omniretarget_original-locomotion`
- Semantic reward config: `null`.
- Object-position termination threshold: `0.25 m` (unchanged baseline).
