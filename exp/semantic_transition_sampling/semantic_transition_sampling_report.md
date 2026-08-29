# Semantic Transition Sampling (first implementation)

## Scope

The previous Semantic KF-Part / KF-Rel / KF-Dyn and fixed-budget semantic
reward code remains in `holosoma.managers.reward.semantic_keyframes` for
negative ablations.  The paper-facing `paper_dr` semantic presets now use the
original `g1_29dof_wbt_reward_w_object` with `semantic_keyframe=None`.  The
sampling modes below change only the reference timestep selected during reset.

## Modes

`MotionConfig.sampling_mode` accepts:

- `original_adaptive`: the existing `AdaptiveTimestepsSampler` path.  Its
  implementation and parameters are unchanged.
- `semantic_uniform`: semantic transition partition with a uniform categorical
  transition distribution.
- `semantic_adaptive`: transition distribution proportional to the empirical
  failure EMA plus the existing `adaptive_uniform_ratio / K` prior.

All semantic modes retain a global uniform fallback with probability
`adaptive_uniform_ratio` (currently the original sampler default `0.1`).  The
remaining probability samples a transition and then samples an integer motion
frame uniformly from `[start_step, target_step)`.  No event-name/body-part
conditionals are used by the sampler.

## Semantic transition construction

`SemanticTransitionSampler` reads only event trigger timestamps.  JSON event
frames are converted to seconds using JSON `fps` (or an explicitly supplied
`semantic_fps`), then mapped with `round(seconds * motion_fps)`, where
`motion_fps` comes from the motion NPZ metadata.  Missing or conflicting FPS
metadata raises an explicit error.  For ordered triggers `e_0, ..., e_K`, the
sampler constructs `K` intervals `[t_{k-1}, t_k)`; the final tail remains
covered by the global fallback.

## Transition statistics

Each environment stores its assigned transition, target step, and a resolved
flag.  A non-timeout termination before the target records one failure.  A
target crossing records one success and does not reset the episode.  A timeout
or motion end before the target is recorded as a configuration error, not a
task-specific failure.  Statistics are updated once per assignment with the
existing `adaptive_alpha`; no semantic-specific temperature, beta, or alpha is
introduced.

Logged metrics include transition probability, failure/success rate, sample
count, failure score, global fallback fraction, normalized entropy, and top-1
transition/probability.

## Reward equivalence

The S0/S1/S2 paper-facing presets share exactly the same original Omni reward
term dictionary.  Semantic sampling does not add a reward term or alter the
`RewardManager` path; therefore for identical simulator state and action,
S0/S1/S2 compute the same reward as `g1_29dof_wbt_reward_w_object`.

## Sampling sanity check

Command used:

```bash
MPLCONFIGDIR=/tmp/mpl-cache XDG_CACHE_HOME=/tmp/cache \
PYTHONPATH=src/holosoma \
conda run -n hssim python diagnostics/semantic_transition_sampling/sampling_sanity.py \
  --semantic-file src/holosoma_retargeting/holosoma_retargeting/demo_data/semantic_keyframes/sub3_largebox_003_semantic_v2.json \
  --motion-fps 50 --motion-steps 325 --samples 10000 \
  --output-dir exp/semantic_transition_sampling
```

Outputs:

- `semantic_sampling_distribution.png`
- `semantic_sampling_distribution.csv`
- `failure_frame_histogram.png`
- `failure_frame_histogram.csv`
- `failure_transition_summary.csv` (transition ID/name, failures, failure rate)

The seven B4 transitions map to motion intervals:

```text
[0,43), [43,50), [50,110), [110,195), [195,267), [267,270), [270,278)
```

The checked-in 10,000-sample artifacts measured global fallback fractions of
0.1013 and 0.0947 for semantic-uniform and semantic-adaptive respectively.
Semantic-uniform transition counts were
`1251, 1294, 1334, 1236, 1298, 1272, 1302`.  For the artificial sanity-check
failure score set on transition 0, semantic-adaptive probabilities were
approximately `[0.9221, 0.0130, ..., 0.0130]`; this is a directional sampler
test, not a training result.

The sanity script has no rollout terminations, so its failure histogram is
zero-filled.  The runtime sampler separately accumulates the same per-frame
and per-transition histograms from real early terminations during training.

## Tests

```bash
PYTHONPATH=src/holosoma conda run -n hssim pytest -q \
  src/holosoma/holosoma/managers/command/tests \
  src/holosoma/holosoma/managers/reward/tests/test_semantic_keyframes.py
```

Result: **30 passed**.

Covered cases include baseline reward/preset equivalence, interval bounds,
FPS mismatch/missing metadata, cold-start uniformity, global fallback, one-shot
success/failure updates, timeout configuration errors, starvation prevention,
and event-name invariance.

## First-round training commands

These are commands only; this implementation does not start the large runs.
All three presets inherit the same PPO, reward, termination, observation, DR,
seed, and B4 reference motion.  Only reset sampling differs.

```bash
python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object-b4-s0-original-adaptive-paper-dr

python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object-b4-s1-semantic-uniform-paper-dr

python src/holosoma/holosoma/train_agent.py \
  exp:g1-29dof-wbt-w-object-b4-s2-semantic-adaptive-paper-dr
```

The prior KF-reward runs remain available through the legacy semantic reward
presets and should be reported as negative ablations, not mixed into S0/S1/S2.
