# Round 4 Final Staged Report

The field-audited `semantic_v2.json` was used unchanged. No VLM regeneration, keyframe edit, criticality edit, lambda tuning, grid search, additional task, or RL was performed.

## Stage A0 — Scale audit

The original Laplacian term is `10 * sum_v ||r_v||²`; full `E_omni` additionally contains nominal tracking, Q-diagonal configuration regularization, and temporal smoothness. The old additive Part term was `10 * mean_part ||r_v||²`, so 38.55x mixed a sum-vs-mean convention mismatch, different residual magnitudes, and non-Laplacian terms. `N_all=115`; no observed performance ratio was used by normalization.

## Stage A1 — Group-Balanced Part

| Method | Ordinary | Global KF | Part | Edge | Local | SQP |
|---|---:|---:|---:|---:|---:|---:|
| Uniform-2 | 21.795258 | 22.527351 | 44.625629 | 62.293678 | 23.244288 | 440 |
| Legacy Semantic Weight | 21.843308 | 22.582962 | 40.917805 | 58.829105 | 22.680418 | 440 |
| Additive Part Only | 21.799616 | 22.548154 | 43.376699 | 61.156440 | 22.983133 | 440 |
| Group-Balanced Additive Part | 22.237927 | 24.365949 | 28.241048 | 50.732206 | 20.777929 | 440 |

Balanced vs U2: Ordinary +2.0310%, Global KF +8.1616%, Part -36.7156%.
Balanced vs Raw Part: Global KF +8.0618%, Part -34.8935%.
Balanced vs Legacy: Global KF +7.8953%, Part -30.9810%.

Official Balanced metrics: penetration duration=0.005102041, max depth=10.958649 mm, foot skating duration=0.000000000, max velocity=0.000000000, contact preservation=0.938775510.

## Stage gate

- Ordinary <= +1%: False (+2.0310%).
- Global KF <= +0.5%: False (+8.1616%).
- Part improvement >= 5%: True (36.7156%).
- Part better than Raw Additive: True.
- **Stage A pass: False**.

Stage A failed because cardinality-only balancing over-prioritized the semantic group, damaged global/ordinary accuracy, and introduced official penetration/contact failures. Per the pre-registered dependency rule, Stages B, C, and D were not run.

## Required questions

- Q1: 38.55x came from multiple factors: primarily incompatible sum-vs-mean aggregation, plus residual-magnitude differences and non-Laplacian terms in full `E_omni`.
- Q2: Part improvement changed from 2.7987% to 36.7156%, but Global KF degraded +8.1616%; the gate failed.
- Q3: Balanced Additive did not form a better Global-vs-Part Pareto point than Legacy because its semantic gain came with an 8.16% Global KF degradation and physical failures.
- Q4–Q11: Not tested because Stage A failed; Edge, Criticality, and Adaptive results would be causally invalid under the protocol.
- Q12: exact cardinality Scale normalization = Drop; raw Additive Part objective = Keep as the safe additive result; Edge = Unproven; VLM Criticality = Unproven; Adaptive computation = Unproven. Legacy Weight remains the strongest completed semantic baseline.
