# Fixed-Budget Semantic Keyframe Reward Report

## Final method

The original Omni WBT reward terms, formulas, sigmas, weights, penalties, PPO settings, randomization, termination, and curriculum remain unchanged. Semantic preference is implemented by redistributing the original positive tracking-reward budget:

\[
R_t=R_{penalty}(t)+\frac{W_{pos}}{W_{pos}+A_t}
\left[R_{base,pos}(t)+A_t r_{sem}(t)\right].
\]

Equivalently,

\[
q_{base}=R_{base,pos}/W_{pos},\quad
\alpha_t=\frac{A_t}{W_{pos}+A_t},
\]

\[
R_{pos,new}=W_{pos}\left[(1-\alpha_t)q_{base}+\alpha_t r_{sem}\right].
\]

There is no manually selected semantic reward coefficient. `sigma_time` defines temporal support, while the pose/dynamics tolerances are inherited directly from the Omni tracking terms.

## Normalized objectives

For semantic bodies \(B_k\), Part fidelity is

\[
r^{part,pos}_k=\frac{1}{|B_k|}\sum_{b\in B_k}
\exp(-\|p_b-p_b^*\|^2/\sigma_{pos}^2),
\]

\[
r^{part,rot}_k=\frac{1}{|B_k|}\sum_{b\in B_k}
\exp(-d_R(q_b,q_b^*)^2/\sigma_{rot}^2),
\]

\[
r^{part}_k=(r^{part,pos}_k+r^{part,rot}_k)/2.
\]

Dynamics fidelity is

\[
r^{dyn,lin}_k=\frac{1}{|B_k|}\sum_b
\exp(-\|v_b-v_b^*\|^2/\sigma_{lin}^2),
\]

\[
r^{dyn,ang}_k=\frac{1}{|B_k|}\sum_b
\exp(-\|\omega_b-\omega_b^*\|^2/\sigma_{ang}^2),
\qquad
r^{dyn}_k=(r^{dyn,lin}_k+r^{dyn,ang}_k)/2.
\]

For each valid body/external-entity edge, Relative Geometry uses the entity-local quantities

\[
\delta p=R_j^T(p_b-p_j),\qquad q_{rel}=q_j^{-1}q_b,
\]

and averages the corresponding normalized position and orientation exponentials across all valid edges. No relation is constructed from an event name.

All pose and relation position terms inherit `sigma_pos=0.3`; all pose and relation orientation terms inherit `sigma_rot=0.4`; dynamics inherits `sigma_lin=1.0` and `sigma_ang=3.14`. These values are read from the active baseline term parameters rather than repeated in `SemanticKeyframeRewardCfg`.

## Temporal and valid-objective aggregation

Every objective shares the transition-truncated gate

\[
g_k(t)=\exp(-(t-t_k)^2/(2\sigma_t^2))
\mathbf 1[t\in window_k]\mathbf 1[t<t_{k+1}],
\]

with semantic and motion frame indices mapped through their timestamps and validated FPS metadata. Per-objective event aggregation is

\[
r_i(t)=\frac{\sum_k valid_{i,k}g_k(t)r_{i,k}(t)}
{\max(\sum_k valid_{i,k}g_k(t),\epsilon)}.
\]

The semantic activity is `A(t)=clamp(sum_k g_k(t), 0, 1)`. Enabled and available objectives form \(V_t\), and

\[
r_{sem}(t)=\frac{1}{|V_t|}\sum_{i\in V_t}r_i(t).
\]

Part, Rel, and Dyn therefore have equal prior importance. If there is no external entity or no valid relation edge, Rel is `INVALID` and is excluded from the denominator. For example, Part `0.8`, invalid Rel, and Dyn `0.6` produce `r_sem=0.7`.

## Positive budget and penalties

`compute_base_positive_reward_budget()` inspects the active reward preset and sums only positive, non-semantic term weights. Current WBT positive terms are bounded exponential tracking terms:

- robot-only: `W_pos = 5.0`;
- robot plus object: `W_pos = 7.0`.

Negative action-rate, joint-limit, and undesired-contact terms are evaluated and accumulated through the original path. They are not multiplied by the semantic budget factor: scaling penalties would alter safety/regularization strength as a function of semantic timing and would no longer reproduce the baseline at fixed state.

The allocation has four direct properties:

1. When `A=0`, `alpha=0`, `R_pos_new=R_base_pos`, and the complete reward equals the baseline.
2. If `R_base_pos=W_pos` and `r_sem=1`, then `R_pos_new=W_pos` for every `A` in `[0,1]`.
3. Semantic reward is a contribution taken from the same envelope, not an added bonus.
4. There are no learned, scheduled, searched, or hand-tuned semantic magnitude parameters.

## Presets and experiments

Final reward preset aliases:

- `g1_29dof_wbt_semantic_reward`
- `g1_29dof_wbt_w_object_semantic_reward`

Controlled experiments:

- R0 `exp:g1-29dof-wbt-w-object-semantic-r0-u2-omni`
- R1 `exp:g1-29dof-wbt-w-object-semantic-r1-b4-omni`
- R2 `exp:g1-29dof-wbt-w-object-semantic-r2-b4-part`
- R3 `exp:g1-29dof-wbt-w-object-semantic-r3-b4-part-rel`
- R4 `exp:g1-29dof-wbt-w-object-semantic-r4-b4-full`

R2/R3/R4 use `enable_part`, `enable_rel`, and `enable_dyn` only as structural ablation switches. Disabled or invalid objectives leave \(V_t\); they do not contribute zero scores and component count cannot change the maximum reward budget.

## Runtime logs

The final runtime records:

- `semantic/activity`, `semantic/alpha`;
- `semantic/part`, `semantic/rel`, `semantic/dyn`, `semantic/combined`;
- `semantic/valid_objective_count`;
- `reward/base_positive`, `reward/base_contribution`;
- `reward/semantic_contribution`, `reward/positive_total`;
- `reward/penalty`, `reward/total`.

The actual contributions obey

\[
base\_contribution=\frac{W_{pos}}{W_{pos}+A}R_{base,pos},
\]

\[
semantic\_contribution=\frac{W_{pos}}{W_{pos}+A}A r_{sem},
\]

and their sum is `reward/positive_total`.

## Reward-budget diagnostic

The 325-frame B4 perfect-reference replay produced:

- automatically inferred `W_pos = 7.0`;
- `A_max = 1.0`;
- `alpha_max = 0.125`, exactly the theoretical `1/(7+1)` bound;
- maximum base contribution `7.0`, minimum at keyframe center `6.125`;
- maximum semantic contribution `0.875`;
- positive-total mean `7.0` and maximum `7.0000009537` (float32, within `1e-6` of the exact budget);
- six exact `A=0` frames, with zero alpha and zero difference from the baseline;
- `positive_budget_not_exceeded = true` under float32 tolerance.

Artifacts:

- `semantic_reward_budget_diagnostic.json`
- `semantic_reward_budget_trace.csv`
- `semantic_reward_budget_diagnostic.png`

They are stored in `src/holosoma_retargeting/holosoma_retargeting/benchmark_results_full_event_transition_truncation/rl/semantic_reward_diagnostic/`.

This diagnostic verifies implementation invariants only; it was not used to select any coefficient.

## Required-question audit

1. Robot-only/object budgets are `5.0/7.0`.
2. They are computed from positive non-semantic baseline term weights.
3. Penalties are not scaled because their regularization strength must remain identical to Omni.
4. Each sub-error is exponentiated to `[0,1]`, then averaged equally.
5. Part/Rel/Dyn use equal prior weight to avoid a replacement set of manual coefficients.
6. Unavailable Rel is invalid and excluded, not inserted as zero.
7. Per-component semantic magnitude coefficients were removed entirely.
8. No global semantic magnitude coefficient was introduced; semantic/base mixing follows only from `W_pos` and `A(t)`.
9. `A=0` restores the baseline numerically and is covered by unit and RewardManager integration tests.
10. The semantic-keyframe maximum positive reward remains mathematically exactly `W_pos`.
11. There is no event-specific reward branch. Event names are diagnostic strings only, and rename invariance is tested through final reward.
12. Focused tests, the full no-simulator suite, and a live IsaacSim integration smoke all pass.

## Validation

- Focused semantic/fixed-budget tests: `21 passed`.
- Full no-simulator suite: `225 passed, 1 skipped, 161 deselected`.
- Live IsaacSim R4 smoke: two environments, one PPO iteration, 48 timesteps, exit code zero; all requested semantic and reward-allocation metrics appeared in the training log.
- Ruff, shell syntax, and `git diff --check`: passed.

No large-scale training, weight search, PPO change, B4 retargeting change, or new reward component was performed.

## Training command

```bash
./scripts/train_semantic_wbt.sh r4
```

For a small integration run:

```bash
NUM_ENVS=64 TRAINING_ITERATIONS=100 ./scripts/train_semantic_wbt.sh r4
```
