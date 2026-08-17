# Semantic Budget Report

This run tests only exact-trigger 2→4 refinement on top of the registered Uniform-2 baseline. Ordinary frames remain configured at two iterations, and all four iterations at a semantic trigger use one unchanged objective.

## Decision

**Semantic Budget: Unproven.**

## Q1–Q10

1. **Legacy compatibility:** zero drift = `True`; max qpos delta 0, metric delta 0 mm, SQP 440.
2. **Added compute:** configured +14 SQP, actual +14 calls, wall +0.926 s.
3. **Semantic quality vs Legacy:** Part Exact +0.329%, Part ±1 +0.100%, Local +0.082%, Edge +0.153% (positive means improvement).
4. **Guardrails:** Global vs U2 +0.263%, Ordinary vs Legacy -0.000%, physical pass vs Legacy = `True`.
5. **Event-wise:** largest Part benefit is `lift` (+0.715%); non-benefiting events: approach, carry_mid, arrive.
6. **Random control:** Semantic Part Exact 40.3344 mm vs random mean 40.4599±0.0376 mm; it beats the random mean on 5/7 events.
7. **Original-Budget control:** Part gain vs U2 +0.619% and vs Legacy -9.592%; compare Semantic-Budget +0.329% vs Legacy.
8. **Complementarity:** extra semantic refinement shows additional gain beyond Legacy weighting on this sequence.
9. **Efficiency:** 0.023520 Part-improvement percentage points per extra actual SQP and 0.355664 per extra wall-second.
10. **Final judgment:** `Unproven` under the registered ordinary/global/physical, ≥1% Part, and random-control criteria.

## Eligibility and causality

All non-frame0 triggers from the deterministic projection are eligible: approach, contact, lift, carry_mid, arrive, place, and release. Random candidates exclude frame 0 and every semantic trigger ±3. No VLM call, criticality, Edge objective, adaptive compute, or later-stage optimizer is used.

The immutable Legacy anchor retains its historical contact/lift/place/release residual support so the required zero-drift test is meaningful. The complete projected event list controls budget eligibility; therefore objective and extra-compute timing are not conflated.
