# Round 4 pre-registered scale stop and field-only-repair rerun

The VLM pipeline regenerated `sub3_largebox_003_semantic_v2.json` from the original video and SMPL-X inputs using the field-only repair policy. Attempt 0 had only `start.body_parts=[]` invalid. Attempt 1 tried to replace `body_parts` in all eight events and was rejected by the local allowlist. Attempt 2 changed only `start.body_parts` to `pelvis` and was accepted. Every other field from attempt 0 remained immutable. Consequently, `contact`, `lift`, and `release` retain `left_hand|right_hand`.

The registered methods were then force-rerun with unchanged parameters. **A3 Additive Part Only** used `sigma=2`, `lambda_part=1`, frame-0 budget 50, and later-frame budget 2:

- Frames: 196
- Actual SQP / solver calls: 440
- Wall time: 28.344466 s
- Optimization time: 23.803313 s
- Mean semantic-active `E_omni`: 0.7814594744
- Mean semantic-active `E_part`: 0.0202713060
- Symmetric magnitude ratio: 38.55003105×

This still exceeds the pre-registered one-order-of-magnitude threshold, so no lambda was changed and the later Round 4 groups remain stopped. The user explicitly requested this registered-baseline rerun after that stop.

The method-independent Part Only diagnostics under the corrected bilateral semantics are:

- Ordinary global error: 21.799616 mm
- Keyframe global exact error: 22.548154 mm
- Semantic part exact error: 43.376699 mm
- Semantic edge exact error: 61.156440 mm
- Semantic local exact error: 22.983133 mm
- All-event VLM-criticality-weighted part exact error: 38.678557 mm
- All-event VLM-criticality-weighted edge exact error: 61.015979 mm

`place` is validly labeled `pelvis`, but the original interaction graph has no cross-entity edge incident to the pelvis at that event. Edge aggregates therefore use the physically evaluable critical events `contact|lift|release`; `place` is explicitly recorded as unavailable rather than assigned a fabricated zero or forcing a semantic relabel.

Against the force-rerun Uniform-2 baseline, Part Only changes Ordinary by +0.0200%, Global KF Exact by +0.0923%, Part Exact by -2.7987%, Edge Exact by -1.8256%, and Local Exact by -1.1235%. It improves all registered semantic metrics while remaining inside the 1% ordinary and 0.5% global guardrails. Official penetration duration/max depth, foot skating duration/max velocity, and contact preservation exactly match Uniform-2 at 0/0/0/0/1.

Against force-rerun Legacy Semantic Weight, Part Only changes Ordinary by -0.2000%, Global KF Exact by -0.1541%, Part Exact by +6.0094%, Edge Exact by +3.9561%, and Local Exact by +1.3347%. Part Only is slightly more stable globally, while Legacy Weight remains stronger on semantic Part/Edge/Local precision. See `registered_baseline_comparison.csv` for all raw values.

No hyperparameter was changed, no additional sequence was run, and no RL was started.
