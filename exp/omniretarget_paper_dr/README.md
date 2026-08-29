# B4 OmniRetarget paper-DR retraining

Primary source: [OmniRetarget paper](https://omniretarget.github.io/static/images/paper.pdf).

The reusable configuration is defined in
`src/holosoma/holosoma/config_values/wbt/g1/paper_dr.py` and registered as a
built-in experiment in `holosoma.config_values.experiment`.  The launcher is
`scripts/train_omniretarget_paper_dr.sh`; this directory contains experiment
records only.

## Implemented alignment

| Area | Paper | This preset |
|---|---|---|
| Object mass | 0.1--2.0 kg | 0.1--2.0 kg final mass |
| Object COM | +/-0.08 m | +/-0.08 m independently on x/y/z |
| Object inertia | 50--150% | 50--150% of nominal Ixx/Iyy/Izz, independent of mass |
| Torso COM | +/-0.025/0.05/0.075 m | exact x/y/z ranges |
| Joint default position | +/-0.01 rad | exact range |
| Random push | 0.3 m/s, 0.78 rad/s, every 1--3 s | exact limits and interval |
| Observation noise | Rot6D 0.05; linear/angular velocity 0.5/0.2; joint position/velocity 0.01/0.5 | exact uniform half-widths |
| Policy observation | reference joint state and pelvis error; pelvis velocity; joint state; previous action | actor terms match this minimal proprioceptive observation (160 dimensions) |

Robot/object material randomization is absent.  PD-gain/RFI and action-delay
perturbations are disabled; their inert state-holder terms remain because the
manager/action paths require those objects.  Initial-pose perturbation is zero.

The formal sampling ablation is split into three registered presets: **S0**
(`original_adaptive`), **S1** (`semantic_uniform`), and **S2**
(`semantic_adaptive`).  S0 preserves the repository's original adaptive
sampler numerically.  S1/S2 change only reset/reference-timestep sampling;
they use the original Omni reward and do not enable the legacy semantic KF
reward.  The older KF reward presets remain available only as explicit
negative ablations.

## Known gaps outside the safely supported preset

1. **Object shape +/-10% is not active.** IsaacLab requires per-environment
   collision scaling before physics startup and `replicate_physics=False`.
   Holosoma currently runs its DR manager after a replicated scene starts.
   Enabling hot scale would produce undefined collision physics; switching off
   replication for 4096 environments is a separate simulator change that needs
   profiling before it can be used for these runs.
2. The paper enables the 1.0 m / 45 degree object termination only after the
   policy has achieved "reasonable body tracking", but does not publish the
   gate or switch iteration.  This preset uses the published thresholds from
   the beginning and retains the repository's body thresholds, whose numerical
   values are also not published by OmniRetarget.
3. The paper groups all box-moving motions into one multi-task policy.  These
   experiments intentionally contain only the available B4 clip, so their SR is
   not a direct reproduction of the paper's grouped-policy result.
4. The semantic transition sampler is an ablation in this repository.  It
   consumes the semantic event JSON only for reset/reference-timestep
   sampling; reward, termination, DR, observations, and PPO settings remain
   those of the original WBT formulation.

## Launch

```bash
NUM_ENVS=4096 TRAINING_ITERATIONS=30000 LOGGER_PRESET=logger:wandb \
  scripts/train_omniretarget_paper_dr.sh s0

NUM_ENVS=4096 TRAINING_ITERATIONS=30000 LOGGER_PRESET=logger:wandb \
  scripts/train_omniretarget_paper_dr.sh s1

NUM_ENVS=4096 TRAINING_ITERATIONS=30000 LOGGER_PRESET=logger:wandb \
  scripts/train_omniretarget_paper_dr.sh s2
```

If running two variants concurrently, use separate tmux windows, for example:

```bash
tmux new-session -d -s b4_paperdr_s42
tmux rename-window -t b4_paperdr_s42:0 s0
tmux new-window -t b4_paperdr_s42 -n s2
# launch s0 in window 0 and s2 in window 1
```
