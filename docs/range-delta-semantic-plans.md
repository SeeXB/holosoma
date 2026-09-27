# Signed change ratios for semantic plans (experimental v1)

The VLM watches the entire video and selects a signal, a comparison reference, a
signed change ratio, crossing direction, and approximate time range for each
action boundary. It still writes the event logic; GT only executes that logic.

For raw signal `s`, reference frame `b`, and frame `t`:

```text
D = max(s over the whole clip) - min(s over the whole clip)
r(t; b) = (s(t) - s(b)) / D
```

This denominator was explicitly selected by the user. A ratio of +0.2 means an
increase of 20% of the **full-clip amplitude**, not 20% of the initial value, a
20th percentile, or 20% of elapsed time. A zero initial value is valid. A constant
signal (`D=0`) is unresolved; no epsilon invents a scale for it.

The existing available signal vocabulary is inherited (66 signals for these OMOMO
bundles). Raw definitions/units remain unchanged; thresholds in generated plans
are dimensionless. Height displacement retains its sign. Absolute vertical speed
does not acquire up/down motion information just by being normalized.

## Rules

```json
{
  "time_range_s": [0.5, 1.5],
  "signal": "object_height",
  "reference": "range_start",
  "ratio": 0.2,
  "crossing": "up"
}
```

Numbers above only illustrate the schema, not a task-specific answer.

| Reference | Comparison frame |
|---|---|
| `clip_start` | First frame of the clip |
| `range_start` | First GT sample inside this boundary's visual time range |
| `action_start` | Resolved start frame of this action; available only to its end rule |

The reference remains fixed throughout a search. A crossing must occur after the
reference frame. An end additionally occurs strictly after its action start.

- `up`: `r[t-1] < ratio <= r[t]`.
- `down`: `r[t-1] > ratio >= r[t]`.

The ratio's sign and crossing direction are **independent**. A signal may fall
through a positive ratio while still remaining above its reference. For
`[0,1,2,3,2,1,0]` with the first frame as reference, ratio +0.5 is crossed upwards
at frame 2 and downwards at frame 5. A speed down-crossing means slowing down,
not downward movement; lifting/lowering instead require a directional position
signal such as object height.

The executor returns the first real matching crossing in the declared interval.
It records all matching candidates, reference/raw values, whole-clip min/max/span,
raw equivalent threshold and ratios before/at the selected frame. No smoothing,
quiet-baseline estimation, physical speed floors, MAD/noise thresholds, dwell,
hysteresis, search padding, forced clipping or nearest-value fallback is used.

If the first action already exists at video start, its start may be the JSON
string `"clip_start"`. If the final action continues at video end, its end may be
`"clip_end"`. These are explicitly censored observations, not GT detections.
An otherwise valid single action with both video-edge boundaries is reported
separately: it exercises no ratio rule and is not evidence of GT localization success.

## Limits that normalization does not solve

- Tiny jitter also becomes a large normalized fraction. The raw amplitude is
  always retained for review; there is no hidden minimum-amplitude hyperparameter.
- An outlier changes the whole-clip amplitude. This v1 does not silently replace
  min/max with quantiles or add an outlier-filtering parameter.
- VLM ratios remain visual estimates. Normalization makes their meaning explicit,
  but does not establish boundary accuracy or fix an incorrectly interpreted action.
- A joint-to-object-origin distance is not a mesh-surface contact distance, and
  kinematics alone do not confirm grasp forces or support contacts.
- Full-clip normalization uses future GT samples; this is an offline annotation
  procedure, not an online event detector.

Whole plans are emitted only if all boundaries are resolved/censored and all event
durations are positive. Rejected plans retain every event in audit files; partial
events are never promoted to an apparently successful single-event plan. Independent
schema errors are returned together in each repair to avoid spending separate API
requests correcting one field at a time.

## Code and execution

Schema: `holosoma.range_delta_plan.v1`.
Implementation: `semantic_keyframes/range_delta.py`.
Existing `execute_dynamic_plan` dispatches this schema and preserves old behavior.

Use Miniconda `hsretargeting`, with the existing full-video preparation manifest:

```bash
python tools/generate_bounded_semantic_plans.py \
  --method range-delta \
  --manifest exp/semantic_plans/regenerate_latest12_bounded_v1_qwen35_20260927/preparation.json \
  --output-root exp/semantic_plans/NEW_OUTPUT_DIRECTORY \
  --expected-host llm-1cbh3bu15hok8ksy.cn-beijing.maas.aliyuncs.com \
  --model qwen3.5-27b --max-workers 2 --max-repairs 3
```

`--prepare-only` makes zero API calls, verifies video/GT alignment and snapshots
source/input hashes. Complete MP4s are sent; there is no client-side API sampling.
The model override is process-local; `.env`, canonical plans and training inputs
are not rewritten. Every model reply and GT resolution is saved separately.
