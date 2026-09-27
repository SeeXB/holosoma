# Adjacent-frame semantic rules: one VLM quantity per action

**Historical run-based experiment, superseded by the user's correction.** The
current task is to locate start and end frames with separate delta predicates;
see [the keyframe method](adjacent-delta-keyframes.md). Do not use the following
continuous-run formulation or its short-run interpretation as the current spec.

This version implements the user's correction: the comparison is always the
previous frame and the current frame. The VLM does not select a reference frame,
frame number, time range, lag, onset/exit pair, or video-edge boundary.

```text
D = max(s over the entire clip) - min(s over the entire clip)
d[t] = (s[t] - s[t-1]) / D, for t = 1,...,N-1
```

The denominator remains the user-approved full-clip raw-signal amplitude. It is
not the amplitude of the differences. Frame zero has no predecessor, so no delta
is fabricated there. For a constant signal, D=0 and normalization is unresolved.

The VLM still selects semantic labels, relevant body regions and a signal, but
its numeric output is exactly one **threshold per action**:

```json
{
  "action": "example_action",
  "body_parts": ["left_hand"],
  "signal": "object_height",
  "condition": "increasing",
  "threshold": 0.01,
  "rationale": "Visible evidence for this action and signal"
}
```

This example explains syntax only; it is not a reference annotation.

| Condition | Per-frame predicate |
|---|---|
| `increasing` | `d[t] >= threshold`, threshold > 0 |
| `decreasing` | `d[t] <= -threshold`, threshold > 0 |
| `steady` | `abs(d[t]) <= threshold`, threshold >= 0 |

Threshold is a nonnegative fraction in [0,1]. The condition supplies the sign.
0.01 means one percent of the signal's full-clip amplitude **per adjacent step**,
not one percent over a whole action. The VLM does not output another threshold for
ending the action. Criticality is also not another VLM number in this experiment;
the compatibility exporter uses a clearly marked uniform placeholder (level 3 /
weight 0.75). These experimental weights are not installed into training data.

## Automatic interval location

1. Evaluate the selected predicate on all adjacent steps.
2. Find its maximal contiguous true runs, with no filtering or minimum duration.
3. In VLM-declared action order, choose the first whole run starting after the
   preceding action's final step. An ongoing run is not cut into a new action.
4. A run on steps a..b corresponds to the pose interval [a-1,b]. Its trigger is a,
   the first current frame with a measured matching delta.
5. If an action cannot resolve, later chronological assignments are marked
   blocked. All candidate runs remain in the audit; partial phases are not exported
   as a successful complete plan.

A one-step run has a positive one-frame duration; it is explicitly flagged as
insufficient evidence for a sustained phase. Multiple candidate runs and a
predicate true throughout the whole video are also flagged. Clip censoring is
inferred from a matched run touching the first/last measured step, not selected
by the VLM.

There is no smoothing, stationary-baseline estimator, noise/velocity floor,
hysteresis, dwell, custom search window, nearest-value fallback or epsilon
denominator. The match order is a deterministic convention, not an assurance that
the earliest numeric run is the correct semantic action.

## Interpretation and limits

- Height delta gives rise/fall directly; distance delta gives approach/separation.
- Speed delta is **change in speed**. A constant nonzero speed has near-zero delta,
  so `steady` on speed does not establish that the object has stopped.
- Cumulative body travel and unsigned rotation have nonnegative deltas. Their
  per-step increments measure movement/rotation amount, not absolute pose or time.
- Small pose jitter can fragment runs or be magnified by a small denominator.
  Raw amplitude, per-step ratios and every candidate run are retained for review.
- Adjacent-frame thresholds depend on the sampling rate. This implementation uses
  the native aligned GT rate without resampling or multiplying the difference by
  fps. In this OMOMO comparison every clip uses 30 Hz.
- Sequential first-run matching cannot guarantee the intended phase when the same
  condition occurs repeatedly. Whole-plan numerical acceptance is not semantic
  accuracy; short-run counts and video/curve review must be reported separately.

## Run

Schema: `holosoma.adjacent_delta_plan.v1`.
Implementation: `semantic_keyframes/adjacent_delta.py`.
The legacy `execute_dynamic_plan` dispatches this schema; old plans remain readable.

In Miniconda `hsretargeting`:

```bash
python tools/generate_bounded_semantic_plans.py \
  --method adjacent-delta \
  --manifest exp/semantic_plans/regenerate_latest12_range_delta_v1_qwen35_20260927/preparation.json \
  --output-root exp/semantic_plans/NEW_OUTPUT_DIRECTORY \
  --expected-host llm-1cbh3bu15hok8ksy.cn-beijing.maas.aliyuncs.com \
  --model qwen3.5-27b --max-workers 2 --max-repairs 3
```

The batch sends full MP4s with no client-side sampling, verifies aligned video/GT,
and records all prompts, replies, candidates and input/source hashes. The model
override is process-local; `.env` and current training plans are unchanged.
