# Constrained Semantic Refinement Report

Fixed experiment: OMOMO/sub3_largebox_003, epsilon_global=0.005, Part-only, exact triggers, at most two extra SQP solves.

## Implemented formulation

Stage 1 uses the unchanged Uniform-2 solver. Stage 2 solves `min sqrt(10/N_part) ||r_part + J_part delta_q||_2` subject to `sqrt(10) ||r_lap + J_lap delta_q||_2 <= sqrt(1.005 E_lap_base)` and the shared original foot, collision, self-collision, joint-limit, and `||delta_q||_2 <= 0.2` constraints. Every candidate is checked with nonlinear FK and falls back to q_base when invalid.

## Q1. Stability and acceptance

Acceptance rate: 0.0000% (0/8); fallback count: 8. Of 16 attempts, 15 were SOCP-optimal and 1 infeasible. Maximum attempted nonlinear budget utilization: 1.005402931.
All 15 solved attempts reached the existing 0.2 step bound. Active original constraint categories observed through solver duals/geometric activity: foot_sticking, joint_limits, nonpenetration, step_bound; the global SOC was also active to numerical precision.

## Q2. Constrained vs Uniform-2

Global KF Exact change: +0.000000%; Part Exact change: +0.000000%; Edge change: +0.000000%; Local change: +0.000000%.

## Q3. Critical event effects

- contact: evaluator Part -0.0000%, rejected-candidate optimization Part +11.1333%, Global +0.0000%, fallback.
- lift: evaluator Part -0.0000%, rejected-candidate optimization Part +18.6846%, Global +0.0000%, fallback.
- place: evaluator Part -0.0000%, rejected-candidate optimization Part +31.7295%, Global +0.0000%, fallback.
- release: evaluator Part -0.0000%, rejected-candidate optimization Part +38.6205%, Global +0.0000%, fallback.

## Q4. Events unable to improve

Fallback events: start, approach, contact, lift, carry_mid, arrive, place, release. The linearized solves found substantial Part descent, but nonlinear global overshoot and/or penetration rejected every candidate; approach iteration 2 was infeasible because its rejected first candidate already lay outside the true global budget.

## Q5. Optimization constraint vs evaluator

Maximum selected nonlinear E_lap change was +0.000000% (required <= +0.5%). Evaluator Global KF Exact changed +0.000000%. Final metrics agree only because every event fell back to U2. At candidate level there is a linearization mismatch: the SOCP global slack is approximately zero while true utilization reaches 1.005402931, so several candidates violate the nonlinear budget by up to 0.540293% of the allowed budget.

## Q6. Comparison with Raw Additive

Raw Additive Part Exact=43.376699 mm; constrained=44.625629 mm.

## Q7. Comparison with Legacy

Legacy (Global, Part)=(22.582962, 40.917805) mm; constrained=(22.527351, 44.625629) mm.

## Q8. Extra compute

Constrained used 456 SQP and 28.987061 s versus U2 440 SQP and 27.753840 s. Mean extra SQP/trigger=2.000000.

## Q9. Official failures

Penetration duration/max depth: 0.0/0.0 mm (U2 0.0/0.0 mm); skating duration/max velocity: 0.0/0.0 (U2 0.0/0.0); contact preservation: 1.0 (U2 1.0).

## Q10. Decision

**Needs Reformulation**. Registered conditions: Ordinary=PASS, Global=PASS, Part>=5%=FAIL, Physical=PASS, Acceptance=FAIL.

## Backward compatibility

The pre-first-refinement base trajectory (including the first eligible trigger) has max abs q difference `0` from standalone Uniform-2.
