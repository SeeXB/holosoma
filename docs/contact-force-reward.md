# Per-part force-shaped contact reward

This opt-in reward combines a small distance-based approach signal with
bounded amplification from current target-object contact forces. It operates
independently on each active semantic key part, without requiring all parts
to contact simultaneously. Existing motion-tracking terms remain unchanged.

## Objective

For key part $k$ at reference time $t$, let $c_{t,k}$ be its robot-local
support point, $a_{t,k}$ the object-local target, $n^o_{t,k}$ the outward
object-surface unit normal, and $\alpha_{t,k}\in[0,1]$ the reference contact
activation. Transform these using the **actual simulated** body and object
poses:

$$
p_{t,k}=x_{t,k}+R_{t,k}c_{t,k},\qquad
q_{t,k}=x_{t,o}+R_{t,o}a_{t,k},\qquad
n_{t,k}=R_{t,o}n^o_{t,k}.
$$

The sensor reports the current target-object-on-body contact-force vector
$F^{o\to k}_{t,k}$. Its nonnegative projection onto the target surface normal
and the proximity kernel are

$$
f_{t,k}=\max(0,n_{t,k}^{\mathsf T}F^{o\to k}_{t,k}),\qquad
s_{t,k}=\exp\left(-\frac{\|p_{t,k}-q_{t,k}\|^2}{\sigma^2}\right).
$$

With $\mathcal A_t=\{k:\alpha_{t,k}>0\}$, the contact reward is

$$
\boxed{
R_t^{\mathrm{contact}}=
\frac{w_c}{\max(1,|\mathcal A_t|)}
\sum_k\alpha_{t,k}s_{t,k}
\left[\eta+(1-\eta)\min\left(\frac{f_{t,k}}{f_s},1\right)\right]
}
$$

Defaults are $w_c=2$, $\sigma=0.05$ m, $\eta=0.1$, and $f_s=4$ N.
The reward manager additionally applies its usual control-step duration.

- With zero force, a part receives only the small approach component.
- Force amplifies only that same part's proximity reward, continuously up to
  $f_s$; another part cannot borrow its force.
- Inactive parts contribute zero. No active parts means zero reward.
- Force above $f_s$ cannot increase the reward. Nonfinite force values do not
  produce a force bonus.

The 4 N value is a tunable saturation scale, initialized from the approximately
3.91 N positive normal-force fifth percentile measured on a historical
White-chair G2 policy. It is not a universal grasp threshold or a calibration
for every task. The force vector is projected onto the reference target's
surface normal, not necessarily the actual collision normal at the current
contact location. A rewarded body contact does not prove force closure or
task success.

## Configuration

In an existing three-group experiment JSON, select the new mode for G3:

```json
{
  "semantic_reward_mode": "contact_force_shaped",
  "contact_file": "/absolute/path/to/reference_contacts_with_normals.npz",
  "contact_sha256": "<SHA256 of contact_file>",
  "contact_approach_fraction": 0.1,
  "contact_force_scale_n": 4.0,
  "contact_object_name": "object"
}
```

These fields augment the existing task/motion/scene/semantic-plan configuration;
they are not a complete standalone experiment specification.
`scripts/run_three_group_experiment.py` selects the reward preset and configures
one object-filtered contact sensor for each distinct body named in the artifact.
`scripts/smoke_contact_force_reward.py` can run an isolated 64-environment,
six-iteration training check from that configuration.

The `contact` mode retains the historical position-only reward. The separate
`contact_force` mode retains the older hard gate requiring every active part
to have at least `minimum_normal_force_n` (default 1 N). The new shaped mode
does not apply that hard gate; force feedback must be explicitly selected.

New Paper-DR policies use the original 154-dimensional actor observations.
Contact forces are used only by the reward; no force channel is added to the
actor. Privileged observations remain available to the critic. Historical
160-dimensional checkpoints retain their serialized configuration and cannot
be directly resumed into a 154-dimensional actor.

## Implementation

- `src/holosoma/holosoma/managers/reward/terms/reference_contact.py`: target
  validation, actual-pose transformations, force projection, reward and metrics.
- `src/holosoma/holosoma/config_values/wbt/g1/reward.py` and
  `src/holosoma/holosoma/config_values/reward.py`: opt-in reward presets/registry.
- `src/holosoma/holosoma/config_types/simulator.py` and
  `src/holosoma/holosoma/simulator/isaacsim/isaacsim.py`: per-body sensors filtered
  to the tracked object's rigid-body prim. They read current `force_matrix_w`,
  not ground/self net force or historical maxima, and reset with environments.
- `scripts/adapt_retarget_contacts_for_rl.py`: transfers and normalizes source
  object normals along with the existing targets and activation. Force modes
  reject artifacts without valid unit normals.
- `src/holosoma/holosoma/config_values/wbt/g1/paper_dr.py`: original actor input.

## Known target-alignment limitation

This release preserves the existing contact-target adapter. Replacing a
rubber-hand support point with a sphere-hand support point does **not** make
them physically equivalent under the same reference pose. The adapter does
not adjust the reference posture or check this alignment.

In the Small-table reference frame 40 audited on 2026-10-14, the left support
points differ by 116.6 mm in the same wrist frame. Distance to the same object
target changes from 5.3 mm for the retargeting support point to 112.1 mm for
the sphere-hand point. The sphere point lies on its mesh; the old rubber
support point lies outside that sphere mesh. This reward release does not
fix that mismatch or reference discontinuities. It should not be presented
as a corrected contact-target-generation algorithm.

## Tests

CPU tests cover approach shaping, force saturation and direction, independent
part activation, invalid forces/parameters, target provenance, normal transfer,
launcher configuration and original actor channels:

```bash
PYTHONPATH=src/holosoma:. python -m pytest -q \
  src/holosoma/holosoma/managers/reward/tests/test_reference_contact.py \
  src/holosoma/holosoma/managers/reward/tests/test_reference_contact_force.py \
  src/holosoma/holosoma/managers/reward/tests/test_reference_contact_force_shaped.py \
  src/holosoma/holosoma/config_values/tests/test_wbt_actor_observations.py \
  scripts/tests/test_adapt_retarget_contacts_for_rl.py
```

Mocked CPU tests do not replace an IsaacSim sensor check or a rollout success
evaluation. Live contact sensing requires the IsaacSim backend.
