# Separate start/end keyframes from adjacent-frame delta

The current v2 format keeps only action, body_parts, start and end in each VLM
action. It requests internal assessment with JSON-only output: no rationale,
criticality, confidence or failure explanations. Signal definitions are grouped
by shared meaning, preserving all 66 available signals and all 18 body regions.

This implements the user's correction to the earlier run-based experiment.
The VLM supplies two independent **boundary rules**, not an action-wide predicate
or a proposed time interval. The executor locates the two keyframes independently.

For each selected raw signal:

```text
D = max(s over the whole clip) - min(s over the whole clip)
d[t] = (s[t] - s[t-1]) / D, t = 1,...,N-1
```

Each of `start` and `end` contains exactly one signal, a condition, and one numeric
threshold. There are therefore two scalar thresholds per action, one per boundary.
Frame numbers, timestamps, search intervals, reference selection, lag, duration,
and a separate action-wide predicate are rejected.

```json
{
  "schema": "holosoma.adjacent_delta_boundary_plan.v2",
  "actions": [{
    "action": "example_action",
    "body_parts": ["<allowed region>"],
    "start": {"signal": "<signal>", "condition": "<condition>", "threshold": "<number>"},
    "end": {"signal": "<signal>", "condition": "<condition>", "threshold": "<number>"}
  }]
}
```

Placeholders describe types, not concrete body selections; replace thresholds
with JSON numbers. `increasing` means
`d[t] >= threshold`, `decreasing` means `d[t] <= -threshold`, and `steady` means
`abs(d[t]) <= threshold`. Increasing/decreasing require a positive threshold;
steady allows zero. Thresholds are dimensionless fractions in [0,1].

The local executor selects the first matching **current frame t** for the start,
strictly after the preceding action's start. It selects the first matching end
frame at or after the current action's start. It does not subtract one from the
selected frame: `[t-1,t]` is the evidence pair for keyframe `t`.

Only the two boundary frames must satisfy their respective predicates. Between
them the signal may rise, pause, fall, or satisfy neither rule. The end is not
derived from the end of a continuous run satisfying the start predicate. There is
also no extra crossing requirement on `d[t-1]`, which would require a third sample.

Start and end may coincide for a point event. A single pair of adjacent samples
is valid boundary evidence; short duration alone is not a failure or a reason to
ask the VLM for a longer event. A matching boundary still does not prove semantic
contact; that requires the signal and video interpretation to be appropriate.

Starts advance in declared order. This convention does not force nonoverlap or
clip an end to the next start. A missing end does not move the next start search
to the clip tail. A missing start blocks later start assignment because the
ordering bound is unknown. No partially resolved plan is exported as complete.

No smoothing, dwell, minimum duration, noise floor, nearest-value fallback, or
epsilon denominator is added. Frame zero has no delta and a constant signal
(`D=0`) is unresolved. Tiny nonzero spans can amplify small raw changes. Native GT
fps is preserved; differences are not multiplied by fps. All candidate frames,
selected pairs, raw spans and measured ratios are saved for review.

V2 exports no rationale or criticality fields and no artificial placeholder
weights. The active retargeter projection and RL sampler do not consume these
fields. The historical precision evaluator still supports weighted metrics;
legacy readers and v1 replay therefore retain their old behavior. The resolved
JSON validator can explicitly omit the historical weight requirement while
retaining body-part and frame validation. Training inputs are not replaced.

Implementation: `semantic_keyframes/adjacent_delta_boundaries.py`; prompt version:
`adjacent_delta_compact_v2`. Both the old v1 boundary schema and archived run
schema remain readable for reproducibility; new generation uses v2:

```bash
python tools/generate_bounded_semantic_plans.py \
  --method adjacent-delta-boundaries \
  --manifest exp/semantic_plans/adjacent_delta_keyframes_v1_qwen35_20260927/preparation.json \
  --output-root exp/semantic_plans/NEW_OUTPUT_DIRECTORY \
  --expected-host llm-1cbh3bu15hok8ksy.cn-beijing.maas.aliyuncs.com \
  --model qwen3.5-27b \
  --max-workers 2 --max-repairs 3
```

Use the existing Miniconda `hsretargeting` environment. The request sends the whole
MP4 without local API frame sampling. Stored plans/prompts/results use a new
experiment directory. Old results retain their original method and interpretation.
