# Semantic Part Parallel Comparison

Fixed task: OMOMO/sub3_largebox_003. Edge, criticality, adaptive compute, sweeps, VLM regeneration, and RL are disabled.

## Q1. Jacobian-Balanced vs Raw Additive

Raw Part improvement=2.798682%; Jacobian-Balanced=31.894030%. Jacobian Global change=+4.383567%, SQP/calls=440/440.

## Q2. Automatic rho

- approach: rho=0.353884.
- contact: rho=0.355614.
- lift: rho=0.358343.
- carry_mid: rho=0.479274.
- arrive: rho=0.337394.
- place: rho=0.339182.
- release: rho=0.342219.

Trigger rho min/max=0.337394/0.479274; there are no near-zero or cap-hit trigger values, and no task-level constant is used. Convention: J_part is the cardinality-matched Part residual Jacobian, and rho multiplies that reference squared objective exactly once. A literal unbalanced Part row subset always has ||J_part||_F <= ||J_lap||_F and would cap rho at 1, degenerating exactly to Raw; the selected convention is value-independent and is recorded explicitly rather than hiding that degeneracy.

## Q3. TR re-solve vs backtracking

Backtracking Part improvement=3.139972%; TR re-solve=3.247191%. TR acceptance=85.71% (6/7).

## Q4. Critical-event trust-region choices

- contact: radius=0.1, alpha=0.125, extra SQP=3.
- lift: radius=0.1, alpha=0.5, extra SQP=2.
- place: radius=0.2, alpha=0.5, extra SQP=1.
- release: radius=0.2, alpha=0.125, extra SQP=3.

## Q5. TR compute

TR uses 456 SQP and 29.467525s; backtracking uses 452 SQP and 29.300561s. Delta=4 SQP, +0.166964s. Relative to U2: +16 SQP.

## Q6. Head-to-head

Jacobian (Global change, Part improvement, SQP)=(+4.383567%, 31.894030%, 440); TR=(+0.028694%, 3.247191%, 456). Physical validity: Jacobian=False, TR=True.

## Q7. Pareto

On the raw two-axis comparison, Jacobian has the better Compute-Part point (440 SQP, 30.392717 mm), while TR has the better Global-Part stability (+0.028694% Global). However, Jacobian is excluded from the registered-feasible Pareto set because it fails Global and physical guardrails. Among registered-feasible methods, TR Global-Part nondominated=yes, Compute-Part nondominated=no; Legacy (440 SQP, 8.308732% Part improvement) dominates TR on the latter axes. Jacobian registered-feasible=no.

## Q8. Decisions

Jacobian Additive: **Drop**. Constrained TR: **Drop**.
Jacobian exceeds the Ordinary/Global guardrails (+1.326947%/+4.383567%) and is not physically valid. TR passes Ordinary, Global, acceptance, and physical checks, but its 3.247191% Part improvement misses the pre-registered 5% threshold.

## Q9. Recommendation

两条新路线均未通过预注册 Keep 条件，不保留 Fast/Precise 双模式。

Candidate-level contact preservation has no frame-local threshold in the existing project; it remains an official whole-trajectory post-hoc guardrail. No new contact threshold was invented.
