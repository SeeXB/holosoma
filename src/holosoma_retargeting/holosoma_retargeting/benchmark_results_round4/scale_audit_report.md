# Round 4 Stage A0 Scale Audit

## Implemented objective forms

- `E_omni` in the prior 38.55x diagnostic is the complete original objective: Laplacian deformation plus nominal tracking, Q-diagonal configuration regularization, and temporal smoothness. Hard constraints are not objective terms.
- The original Laplacian component is `E_lap = 10 * sum_v ||r_v||^2`, a squared sum over all interaction-mesh vertices and XYZ components.
- The prior raw additive implementation reports `E_part = 10 * mean_{v in part} ||r_v||^2`. It therefore does not use the same aggregation convention as `E_lap`.
- For a convention-matched semantic raw term, `E_part_raw = 10 * sum_{v in part} ||r_v||^2`. Squared-sum group balancing is then `(N_all/N_part) * E_part_raw`.

## Observed scale

- Stored semantic-active mean `E_omni`: 0.7814594744.
- Stored semantic-active mean prior mean-aggregated `E_part`: 0.0202713060.
- Previously reported ratio: 38.55003105x.
- Trigger-event mean recomputed `E_lap`: 0.7913748323.
- Trigger-event mean convention-matched raw `E_part`: 0.0553843268.
- Trigger-event mean counterfactual balanced `E_part`: 2.4780958690.

## Attribution

`N_all` is 115. The prior 38.55x number is not a clean geometry-scale ratio: its numerator includes non-Laplacian regularizers while its denominator is a per-part mean. If per-vertex residual magnitudes were equal, that sum-vs-mean mismatch alone would predict approximately `N_all=115` rather than `N_all/N_part`. Event rows show the remaining deviation comes from semantic vertices having different residual magnitudes and from the non-Laplacian terms inside `E_omni`. The new normalization uses only `E_lap`'s squared-sum convention and cardinality; it does not use 38.55 or task performance.

## Gradient diagnostic

The completed raw run did not serialize the linearized `J_L`, so its warm-start Jacobian norms cannot be reconstructed exactly from the trajectory NPZ. Stage A1 records Laplacian/raw-Part/balanced-Part Frobenius norms directly from the existing linearized solver without changing the solve.

## Stage A1 solver-side scale check

- Mean Laplacian residual-Jacobian Frobenius norm: 22.2354827578.
- Mean raw-Part residual-Jacobian Frobenius norm: 7.6182513153.
- Mean balanced-Part residual-Jacobian Frobenius norm: 53.2032715563.
- Mean balanced `E_part`: 1.0546004058; mean complete `E_omni`: 0.9478934074.
- Cardinality balancing removes the value-scale mismatch, but makes the balanced Part residual Jacobian substantially larger than the full Laplacian residual Jacobian. The performance gate, not a tuned coefficient, decides whether this exact normalization is retained.
