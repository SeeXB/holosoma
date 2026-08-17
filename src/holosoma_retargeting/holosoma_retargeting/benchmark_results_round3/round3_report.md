# Round 3 Semantic Precision Report

Frame partition: ordinary=169, K3=26, exact=4.
All precision errors use the same unweighted uniform-Laplacian residual and are reported in mm.

## Q1 — Uniform computation

Original used 1892 actual SQP / solver calls.
Uniform-1 used 245, reduced 1647 (87.05%), achieved 6.82x speedup, and increased ordinary error by +2.025%.
Uniform-2 used 440, reduced 1452 (76.74%), achieved 4.04x speedup, and changed ordinary error by +0.051%.

## Q2 — Is Uniform-2 stable?

Yes. Ordinary delta vs Original=+0.051%; official penetration duration=0.0000, skating duration=0.0000, contact preservation=1.0000; SQP reduction=76.74% and speedup=4.04x.

## Q3 — Semantic Weight vs Uniform-2

Ordinary=+0.286% (matched ±2%: True); KF exact=+0.333% (improved: False); KF ±1=+0.314%; Part exact=-10.403% (improved: True); Part ±1=-9.249%; auxiliary Local exact=-2.050% (improved: True).
Semantic Weight improves the registered body-part and auxiliary local metrics, but not the registered global keyframe metric.

## Q4 — Semantic Weight vs Original

Ordinary is matched (+0.338%). KF exact is not better (+0.226%), while Part exact changes by -9.883% and Local exact by -2.037%. SQP falls by 76.74% and wall time by 74.99% (4.00x). The complete Q4 target is not met because global keyframe precision did not improve.

## Q5 — Causal preference controls

vs Always-Hand-Matched: ordinary=+0.103%, KF exact=+0.161%, Part exact=-6.685%, SQP delta=+0.000%.
vs Random-Time mean: ordinary=+0.015%, KF exact=+0.079%, Part exact=-6.852%, SQP delta=+0.000%.
vs Wrong-Body: ordinary=+0.114%, KF exact=-0.223%, Part exact=-11.793%, SQP delta=+0.000%.

All controls and Semantic Weight use 440 solver calls and the same topology/kernel/body multipliers. Semantic Weight L1=1747.268869; Random target/matched=1747.268869/1747.268869; Always-Hand target/actual=1747.268869/1747.268869; Wrong-Body target/matched=1747.268869/1747.268869.
Part exact is better than every causal preference control. Global exact is worse than Always-Hand-Matched and Random-Time mean, but better than Wrong-Body; the advantage is body-part-specific rather than global.

## Q6 — Budget 10/4/2 vs 10/4/2/1

New vs U2: ordinary=-0.001%, KF exact=+0.002%, Part exact=-0.497%, SQP=505 (+14.773%).
New vs old: ordinary=-1.660%, KF exact=+0.167%, Part exact=+0.127%, SQP delta=+19.668%.
Raising ordinary frames from one to two iterations removes the old ordinary-frame loss, so ordinary=1 was the main cause of that guardrail weakness. It does not create meaningful semantic precision gains, so it was not the main cause of the budget method's weak semantic result.

## Q7 — Weight + Budget vs Weight

Extra SQP=76 (+17.27%); wall delta=+4.697s (+17.54%); KF exact=-0.038%; Part exact=-0.680%; Local exact=-0.069%.
The extra compute is not justified: 17.27% more SQP buys less than 0.7% on Part and less than 0.1% on KF/Local.

## Q8 — Official OmniRetarget guardrails

No degradation for Semantic Weight vs U2: penetration duration=+0.0000, max depth=+0.000mm; skating duration=+0.0000, max velocity=+0.000000; contact preservation=+0.0000. Exact guardrail match: True.
Budget-Base2 and Weight+Budget also have penetration/skating durations 0 and contact preservation 1. Uniform-1 has penetration duration 0.0255, max depth 11.191mm, and contact preservation 0.9847; the old budget has penetration duration 0.0255 and max depth 11.191mm.

## Q9 — Quality/compute Pareto

Part/SQP Pareto: Semantic Weight, Weight + Budget-Base2. Global-keyframe/wall-time Pareto: Uniform-2.
Among the four registered candidates, Semantic Weight dominates Uniform-2 and Budget-Base2 on Part/SQP, while Weight+Budget buys only a small extra Part gain. Uniform-2 dominates all semantic variants on Global-keyframe/wall-time.

## Pre-registered conditions

A Ordinary preservation: True.
B Semantic precision: False (KF exact +0.333%, Part exact -10.403%).
C Efficiency: True (SQP reduction 76.74%, speedup 4.00x).
D Physical quality: True (penetration/skating/contact exactly match U2).

The full pre-registered core hypothesis is not established because Condition B requires both global keyframe and body-part precision to improve.
Formal final recommendation: retain none under the pre-registered acceptance rule. If the objective is narrowed to semantic body-part/local precision, Weight only is the strongest diagnostic candidate; Budget is not retained.

No parameters were tuned, no additional task was run, and no RL was started.
