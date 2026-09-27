# Full-video phases with bounded GT detection (experimental v1)

The VLM receives the complete aligned MP4, its duration, the signal vocabulary,
and generic anatomy definitions. It selects phase names, body regions, criticality,
and a **time range + signal + transition** for each start/end. It does not see old
plans, task-specific answers, GT curves, physical thresholds, or quantile values.
No client-side video sampling is used. Provider-side sampling is unknown.

Implementation: `semantic_keyframes/bounded_signals.py`, schema
`holosoma.bounded_signal_plan.v1`. Existing quantile and localized-phase schemas
retain their execution behavior. `execute_dynamic_plan` dispatches the new schema.

## Boundary contract

| Signals | Transitions | GT feature |
|---|---|---|
| Height, origin distance, displacement, progress | increase_onset/offset, decrease_onset/offset | Signed derivative |
| Accumulated rotation, accumulated body travel | increase_onset/offset | Signed derivative; no decreasing operation |
| Unsigned linear/angular speeds | motion_onset/offset | Speed itself, not its derivative |
| Explicitly ongoing at video edge | clip_start/end; signal=null | Censored observation, **not** a detected transition |

The existing body signals are retained. Origin distances are not surface contact
measurements; the minimal official-PT-derived bundles lack mesh/contact-force data.
No contact force or support-surface verification is inferred from these signals.

Visual intervals are given in seconds and bounded to the GT timeline. Their width
is at most `min(2.0, max(0.8, 0.4 * last_sample_time))`. The first phase may explicitly
start at frame 0; the last may explicitly end at the last frame. Such censored
boundaries are counted separately. They must use exact edge intervals and null
signals. Overlapping phases are allowed; action start order must be preserved.

## Local detector

1. Median filter, then moving average (0.10 s, rounded to an odd frame count).
2. Use a signed derivative for positional signals; raw smoothed speed for speed
   signals. Reverse sign for decreasing motion.
3. In the visual range plus 0.75 s context on each side, select the 0.20 s segment
   with the smallest median absolute feature value. Verify it is near zero using
   a fixed resolution floor and locally estimated residual noise. A continuous
   nonzero trend cannot be reinterpreted as a stationary baseline.
4. Let `sigma = 1.4826 * MAD`, using the larger of quiet-segment and residual noise.
   Enter threshold is `max(floor, abs(baseline_median) + 4 * sigma)`; exit threshold
   is half of it. Floors are 0.02 m/s, 0.02 rad/s, or 0.005/s for normalized progress.
5. Require 0.10 s of consistent evidence (at least two samples) to establish and
   change the hysteresis state. Only transitions **inside** the visual interval
   may be returned; already-active initial states do not invent an onset.
6. If several real transitions qualify, select the one nearest the visual interval
   midpoint and retain every candidate in the audit. This is **not** a fallback:
   zero matching transitions returns unresolved with frame=null.

All constants are shared across tasks and recorded. These are **provisional
engineering defaults**, not empirically calibrated accuracy guarantees. Slow
continuous motion, changing directions without a quiet baseline, oscillatory body
motion, and inaccurate visual time ranges may remain unresolved. Physical onset
accuracy has not been established against independently annotated frames.

## Output and failure policy

Each request saves prompt, raw response, request metadata, validated plan (if any),
and complete resolution diagnostics. Diagnostics include search/context ranges,
quiet baseline, noise, thresholds, all candidate frames, chosen frame and status.
Schema errors and unsupported GT boundaries may be repaired at most three times.
The detector is not changed per task or tuned after seeing results in a batch.

`plans/omomo/<task>/semantic_plan.json` is emitted only when **every** boundary is
GT-resolved or explicitly clip-censored and all event durations are positive.
Failures keep all actions in audit files; partial resolved events are diagnostics,
not exported pruned plans. No nearest-value fallback, forced endpoint clipping,
or automatic replacement of training inputs is performed.

## Reproduce the experiment

Use the Miniconda `hsretargeting` environment (plus FFmpeg on PATH, or the sibling
`hssim` environment's FFmpeg). The input manifest is the existing preparation JSON
containing task, video/bundle paths, frame rates/counts and SHA256 values.

```bash
python tools/generate_bounded_semantic_plans.py \
  --manifest exp/semantic_plans/regenerate_latest12_threshold_v3_qwen35_20260927/video_input/preparation.json \
  --output-root exp/semantic_plans/NEW_OUTPUT_DIRECTORY \
  --expected-host llm-1cbh3bu15hok8ksy.cn-beijing.maas.aliyuncs.com \
  --model qwen3.5-27b --max-workers 2 --max-repairs 3
```

`--prepare-only` verifies all videos decode and match GT frames/fps, snapshots
source/configuration and hashes protected inputs, and makes zero API requests.
`--tasks` restricts a batch. Already attempted tasks are rejected to avoid hidden
retries. The model override is process-local; the credential file is never edited.
No GPU or RL training process is used by this generator.
