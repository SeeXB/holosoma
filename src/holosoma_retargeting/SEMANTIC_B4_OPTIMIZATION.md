# Semantic B4 Retargeting Optimization

This document specifies the original Semantic B4 retargeting optimization.
It deliberately excludes all later force, affordance, explicit contact-point,
contact-normal, and contact-distance objectives.

## 1. Method in one equation

Semantic B4 retains OmniRetarget's interaction-mesh representation and
physical feasibility constraints. Its two changes are highlighted below:

1. causally reweight the existing Laplacian residuals using semantic events;
2. assign extra SQP iterations only to exact semantic trigger frames.

For trajectory frame \(t\) and SQP iteration \(k\), Semantic B4 solves

$$
\boxed{
\begin{aligned}
\Delta q_{t}^{(k)\star}
={}&
\underset{\Delta q\in\mathbb R^{n_q}}{\arg\min}
\quad
\underbrace{
\lambda_{\mathrm{lap}}
\sum_{i=1}^{V}
\alpha_{t,i}^{\mathrm{B4}}
\left\|
\delta_{t,i}^{R,(k)}
+J_{\delta,t,i}^{(k)}\Delta q
-\delta_{t,i}^{H}
\right\|_2^2
}_{\text{Semantic B4: semantic reweighting of the OmniRetarget interaction residual}}
\\[-1mm]
&\qquad+
\underbrace{
\lambda_{\mathrm{smooth}}
\left\|
q_t^{(k)}+\Delta q-q_{t-1}^{\star}
\right\|_2^2
+
\left\|
Q^{1/2}\left(q_t^{(k)}+\Delta q\right)
\right\|_2^2
}_{\text{simplified OmniRetarget smoothness and posture regularization}}
\\[2mm]
\mathrm{s.t.}\quad
&
\underbrace{
\phi_c\!\left(q_t^{(k)}\right)
+J_{\phi,c}^{(k)}\Delta q
\ge -\epsilon,
\quad \forall c\in\mathcal C_t^{(k)}
}_{\text{robot--object and robot--ground non-penetration}},
\\
&
\underbrace{
\left\|
p_{f,xy}\!\left(q_t^{(k)}\right)
+J_{f,xy}^{(k)}\Delta q
-p_{f,xy}\!\left(q_{t-1}^{\star}\right)
\right\|_{\infty}
\le\eta,
\quad \forall f\in\mathcal F_t
}_{\text{support-foot sticking}},
\\
&
\underbrace{
q^{\min}\le q_t^{(k)}+\Delta q\le q^{\max},
\qquad
\|\Delta q\|_2\le r
}_{\text{joint limits and SQP trust region}},
\\[3mm]
&
q_t^{(k+1)}
=
\operatorname{NormalizeQuat}
\left(q_t^{(k)}+\Delta q_t^{(k)\star}\right),
\qquad
k=0,\ldots,B_t^{\mathrm{B4}}-1,
\\[3mm]
&
\underbrace{
B_t^{\mathrm{B4}}
=
\begin{cases}
50,&t=0,\\
4,&t\in\mathcal T_{\mathrm{sem}},\\
2,&\text{otherwise},
\end{cases}
}_{\text{Semantic B4: compute is added only at exact semantic triggers}},
\\[4mm]
&
\underbrace{
\alpha_{t,i}^{\mathrm{B4}}
=
\frac{\widetilde\alpha_{t,i}}
{\displaystyle\frac{1}{V}\sum_{j=1}^{V}\widetilde\alpha_{t,j}}
}_{\text{mean normalization preserves the per-frame weight scale}},
\\[3mm]
&
\widetilde\alpha_{t,i}
=
\max\left\{
1,
\max_{\substack{e\in\mathcal E\\p\in\mathcal P_e}}
\left[
1+\gamma_e(t)\left(m_{p,i}-1\right)
\right]
\right\},
\\[3mm]
&
m_{p,i}
=
\begin{cases}
m_p,&i=b(p),\\
m_O,&i\in\mathcal N_O\!\left(b(p)\right),\\
1,&\text{otherwise},
\end{cases}
\qquad
m_p=
\begin{cases}
4,&p\in\{\text{left hand},\text{right hand}\},\\
2,&p\in\{\text{pelvis, feet, torso, waist, hips, knees,}\\
&\qquad\text{ankles, shoulders, elbows, wrists}\},
\end{cases}
\qquad
m_O=2,
\\[4mm]
&
\underbrace{
\gamma_e(t)
=
\mathbf 1\!\left[
c_e\le t\le\min\!\left(b_e,c_e^{\mathrm{next}}-1\right)
\right]
\max\left\{
\exp\!\left[-\frac{(t-c_e)^2}{2\sigma^2}\right],
\rho
\right\},
\quad \sigma=2,\ \rho=0.25
}_{\text{Semantic B4: causal, transition-truncated semantic temporal kernel}}.
\end{aligned}
}
$$

In compact form,

$$
\boxed{
\text{Semantic B4}
=
\text{OmniRetarget}
+\text{semantic residual allocation}
+\text{semantic compute allocation}
}
$$

Semantic B4 does not introduce a new contact loss. It reallocates the
importance of the existing OmniRetarget residuals and the number of SQP
iterations used to minimize them.

## 2. Interaction mesh and Laplacian residual

At each frame, the source interaction mesh contains \(N_H\) mapped human
joints and \(N_O\) sampled object points. All human points are expressed in
the source object frame:

$$
\bar h_{t,i}
=R_{O,t}^{H\top}\left(h_{t,i}-p_{O,t}^{H}\right).
$$

The source vertices are

$$
S_t=
\begin{bmatrix}
\bar h_{t,1} & \cdots & \bar h_{t,N_H} &
o_1^H & \cdots & o_{N_O}^H
\end{bmatrix}^{\!\top}
\in\mathbb R^{V\times3},
\qquad V=N_H+N_O.
$$

A Delaunay tetrahedralization of \(S_t\) defines the neighbor set
\(\mathcal N_t(i)\). The uniform Laplacian is

$$
L_{t,ij}=
\begin{cases}
1,&j=i,\\
-1/|\mathcal N_t(i)|,&j\in\mathcal N_t(i),\\
0,&\text{otherwise}.
\end{cases}
$$

The source Laplacian coordinate of vertex \(i\) is therefore

$$
\delta_{t,i}^{H}
=s_{t,i}
-\frac{1}{|\mathcal N_t(i)|}
\sum_{j\in\mathcal N_t(i)}s_{t,j}.
$$

The corresponding robot point in the target object frame is

$$
r_{t,i}(q)
=R_{O,t}^{R\top}\left(p_i^R(q)-p_{O,t}^{R}\right).
$$

The target mesh combines these robot points with the fixed target-object
samples. At the current SQP iterate, its Laplacian coordinates are linearized
as

$$
\delta_t^R\!\left(q_t^{(k)}+\Delta q\right)
\approx
\delta_t^{R,(k)}+J_{\delta,t}^{(k)}\Delta q,
$$

where

$$
J_{\delta,t}^{(k)}
=(L_t\otimes I_3)J_{X,t}^{(k)}.
$$

Thus the residual emphasized by Semantic B4 is

$$
e_{t,i}^{(k)}(\Delta q)
=
\delta_{t,i}^{R,(k)}
+J_{\delta,t,i}^{(k)}\Delta q
-\delta_{t,i}^{H}.
$$

It measures the difference between the updated robot--object spatial
relationship and the corresponding source human--object spatial relationship.

## 3. Semantic residual allocation

Let \(\mathcal E\) be all events in the semantic plan and \(\mathcal P_e\) the
body parts assigned to event \(e\). The active-plan projection deliberately
sets every event confidence to one; criticality values are not used by B4.

For each event:

- \(c_e\) is its trigger frame;
- \(b_e\) is its end frame;
- \(c_e^{\mathrm{next}}\) is the next greater event trigger, when one exists;
- \(\gamma_e(t)\) is its causal temporal strength.

The support starts at the trigger, ends at the event boundary, and is also
truncated immediately before the next semantic transition. Inside that
support, a Gaussian centered on the trigger is lower-bounded by the phase
floor \(\rho=0.25\). There is no pre-trigger Gaussian weighting.

For a body part \(p\), \(b(p)\) is its mapped human vertex and
\(\mathcal N_O(b(p))\) contains the directly adjacent object vertices in the
interaction mesh. B4 boosts the selected body residual and the existing
object-neighbor residuals. Overlapping contributions use `max`, not addition.

The raw weights are divided by their frame mean. Consequently,

$$
\frac{1}{V}\sum_{i=1}^{V}\alpha_{t,i}^{\mathrm{B4}}=1,
$$

so B4 changes relative attention without changing the average Laplacian
weight from frame to frame.

## 4. Simplified OmniRetarget terms and constraints

The constants used by the implementation are

$$
\lambda_{\mathrm{lap}}=10,
\qquad
\lambda_{\mathrm{smooth}}=0.2,
\qquad
\epsilon=0.001\ \mathrm m,
\qquad
\eta=0.001\ \mathrm m,
\qquad
r=0.2.
$$

Here:

- \(q_t^{(k)}\) is the robot configuration at the current SQP iterate;
- \(\Delta q\) is the optimized configuration increment;
- \(q_{t-1}^{\star}\) is the accepted configuration from the previous frame;
- \(Q\) is the posture-regularization matrix;
- \(\phi_c\) is the signed distance for collision pair \(c\), positive when
  separated and negative when penetrating;
- \(J_{\phi,c}^{(k)}\) is the signed-distance Jacobian;
- \(\mathcal C_t^{(k)}\) contains current object/ground collision candidates;
- \(\mathcal F_t\) contains support-foot points classified as sticking;
- \(J_{f,xy}^{(k)}\) is a support-point XY Jacobian;
- \(q^{\min},q^{\max}\) are the model and manually overridden joint limits.

For the G1 configuration, \(q\in\mathbb R^{36}\): seven floating-base qpos
coordinates plus 29 robot joints. The object pose is locked and is not part of
\(\Delta q\). The nonzero entries of \(Q\) are the waist-yaw and waist-roll
weights, both equal to 0.2. A non-augmented original retarget has no nominal
trajectory, so its optional nominal-tracking objective is exactly zero.

The trust-region constraint makes each subproblem a convex quadratic problem
with a second-order-cone constraint. After each solve, the floating-base
quaternion is normalized and all kinematics, Jacobians, Laplacian coordinates,
and signed distances are recomputed for the next SQP iteration.

## 5. Semantic compute allocation

Let \(\mathcal T_{\mathrm{sem}}\) be the set of exact event trigger frames.
Semantic B4 uses a Uniform-2 base and raises only those exact trigger frames to
a four-iteration cap. Frame zero retains the initialization cap of 50.

The solver may terminate before a cap when consecutive objective values satisfy
its numerical convergence check. No neighboring frame receives extra SQP work
solely because it lies inside a semantic event window: temporal semantic
weighting can remain active while the compute budget stays at two.

## 6. Concrete `sub7_smallbox_023` instance

For this sequence,

$$
N_H=15,
\qquad N_O=100,
\qquad V=115,
\qquad
\mathcal T_{\mathrm{sem}}=\{25,49\}.
$$

The two semantic events are

$$
e_0:\quad [25,40],\quad c_0=25,
$$

$$
e_1:\quad [49,99],\quad c_1=49.
$$

Both events select the left and right hands. At an exact trigger,
\(\gamma_e=1\), so the raw hand and neighboring-object weights are

$$
\widetilde\alpha_{\mathrm{hand}}=4,
\qquad
\widetilde\alpha_{\mathrm{object\ neighbor}}=2.
$$

On the temporal plateau, \(\gamma_e=0.25\), giving

$$
\widetilde\alpha_{\mathrm{hand}}=1.75,
\qquad
\widetilde\alpha_{\mathrm{object\ neighbor}}=1.25.
$$

These values are subsequently mean-normalized over all 115 vertices.

The configured iteration caps are

$$
B_t^{\mathrm{B4}}=
\begin{cases}
50,&t=0,\\
4,&t\in\{25,49\},\\
2,&\text{otherwise}.
\end{cases}
$$

Frame zero converged in 11 iterations in the measured canonical run; the two
trigger frames used four iterations and the remaining 141 frames used two.

Because the final semantic event ends at frame 99, the highest-object frame
107 has

$$
\alpha_{107,i}^{\mathrm{B4}}=1,
\quad \forall i,
\qquad
B_{107}^{\mathrm{B4}}=2.
$$

Old Semantic B4 therefore has neither semantic emphasis nor an explicit
hand--object contact objective at frame 107. Its non-penetration constraint can
prevent excessive overlap, but it cannot prevent the hands from separating
from the box.

## 7. Explicitly excluded objectives

For the method defined here,

$$
E_{\mathrm{contact}}
=E_{\mathrm{normal}}
=E_{\mathrm{force}}
=E_{\mathrm{affordance}}
=0.
$$

There is no hand--box zero-gap constraint, palm-normal objective, force-closure
condition, or separate semantic edge objective. Object-neighbor weighting only
changes the importance of existing Laplacian residuals.

## 8. Implementation references

- Per-iteration SQP objective and constraints:
  `holosoma_retargeting/src/interaction_mesh_retargeter.py`
- Semantic temporal and vertex-weight construction:
  `holosoma_retargeting/semantic_keyframes/runtime.py`
- Registered Semantic B4 parameters and schedule:
  `holosoma_retargeting/config_types/semantic.py`
