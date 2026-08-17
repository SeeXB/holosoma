# Final semantic retargeting report

## Compatibility

Cleanup compatibility: **PASS**. Original, Uniform-2, and Legacy trajectories, actual SQP counts, method-independent metrics, and official physical metrics were checked against the preserved Round-4 caches.

Legacy anchor vs Uniform-2: Ordinary `+0.220%`, Global KF `+0.247%`, Part improvement `8.309%`, SQP `440`.

## Sequential decisions

- Semantic Edge: **Drop**. Part+Edge vs Part changed Edge by `+0.000%`, Local by `+0.000%`, and Part by `+0.000%`. The implementation de-duplicates overlap with Legacy object-neighbor weighting.
- VLM Criticality: **Unproven/Drop**. It was required to beat both activation-matched uniform criticality and the five-seed shuffled mean on original-criticality-weighted semantic metrics without violating ordinary/global/physical guardrails.
- Adaptive Compute: **Drop**. Selected adaptive SQP `440` vs 440 and wall time `35.139s` vs fixed semantic `29.991s`; Part change vs fixed `+0.000%`.

## Final method

Selected mode: **`uniform2_semantic_weight_uniform`**. This uses uniform criticality (`c=1`); the immutable historical `uniform2_semantic_weight` mode remains available as the compatibility anchor. The activation-matched control used `c=0.805582`.

The only optimization mechanism retained is mean-one Legacy residual reweighting, with optional modules included only when their registered gate above passes. No additive objective, constrained second stage, backtracking, Jacobian balancing, or trust-region re-solve is used.
