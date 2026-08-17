# Constrained Refinement + Backtracking Report

Fixed protocol: epsilon=0.005, unchanged Part-only SOCP, unchanged 0.2 trust region, exact nonzero triggers, alpha=[1, 0.5, 0.25, 0.125, 0.0625].

## Q1. Acceptance

Previous full-step acceptance was 0/8. Frame 0 is excluded here, leaving 7 eligible triggers. Backtracking acceptance is 5/7 (71.43%).

## Q2-Q3. Event alpha and semantic-gain retention

- approach: first global-pass alpha=0.5, accepted alpha=None, full gain=30.4827%, final gain=0.0000%, retention=0.00%, global=+0.0000%, fallback=True.
- contact: first global-pass alpha=0.5, accepted alpha=0.0625, full gain=4.5125%, final gain=0.4209%, retention=9.33%, global=-0.0101%, fallback=False.
- lift: first global-pass alpha=0.5, accepted alpha=0.0625, full gain=18.9407%, final gain=6.8939%, retention=36.40%, global=-0.0591%, fallback=False.
- carry_mid: first global-pass alpha=0.5, accepted alpha=0.5, full gain=27.8393%, final gain=22.8624%, retention=82.12%, global=+0.1739%, fallback=False.
- arrive: first global-pass alpha=1.0, accepted alpha=None, full gain=30.4597%, final gain=0.0000%, retention=0.00%, global=+0.0000%, fallback=True.
- place: first global-pass alpha=0.5, accepted alpha=0.125, full gain=29.9767%, final gain=19.0583%, retention=63.58%, global=+0.1020%, fallback=False.
- release: first global-pass alpha=1.0, accepted alpha=0.125, full gain=32.2801%, final gain=4.7870%, retention=14.83%, global=-0.0057%, fallback=False.

Mean/median gain retention among accepted events: 41.25% / 36.40%.

## Q4-Q5. Method-independent Part and Global

Versus U2, Part Exact improves 3.139972% and Global KF Exact changes +0.019842% (guardrail <= +0.5%). Ordinary changes -0.001813%.

## Q6. Official physical metrics

Backtracking penetration duration/depth=0.0/0.0 mm, skating duration/max velocity=0.0/0.0, contact=1.0; physical guardrail=PASS.

## Q7. Compute

Backtracking uses 452 SQP and 29.300561s; U2 uses 440 SQP and 27.753840s. Extra=12 SQP and +1.546721s (+5.57%).

## Q8-Q9. Raw Additive and Legacy

Raw Additive (Global, Part)=(22.548154, 43.376699) mm; Legacy=(22.582962, 40.917805) mm; Backtracking=(22.531821, 43.224396) mm.
Backtracking's Part improvement over U2 is 3.139972% versus Raw Additive's 2.798682%: stronger by +0.341290 percentage points, while also having lower Global error than Raw Additive.
Legacy improves Part by 8.308732% at Global change +0.246862%; Backtracking improves Part by 3.139972% at +0.019842%. Neither strictly dominates: Legacy has stronger Part accuracy, while Backtracking preserves Global accuracy better.
Historical diagnostics only (not rerun): Balanced Additive (Global, Part)=(24.365949, 28.241048) mm; previous full-step constrained=(22.527351, 44.625629) mm.

## Q10. Remaining rejection cause

Most frequent alpha-level rejection reason: penetration_violation (36 occurrences), versus 9 global-budget rejections. Fallback events: approach, arrive.

## Q11. Decision

**Needs Trust-Region Reformulation**. Ordinary=PASS, Global=PASS, Part>=5%=FAIL, Physical=PASS, Acceptance>=50%=PASS.

Backtracking alone is insufficient; the diagnosed next formulation is adaptive trust-region contraction plus re-solve, but it is not implemented in this round.

## Backward compatibility

Maximum absolute difference between every recorded Stage-1 q_base and standalone U2 is 0.
