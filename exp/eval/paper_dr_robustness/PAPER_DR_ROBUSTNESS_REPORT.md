# Paper-DR robustness evaluation

Protocol: fixed B4 reference, seed 42, 32 parallel environments, first 10 completed episodes per environment (320 scored episodes), 324 valid motion frames per clip. Object termination thresholds are 1.0 m and 45 degrees. Paper push perturbations are enabled during evaluation; object shape ±10% remains unsupported by the current simulator setup. Success is a timeout with no tracking flag on the final valid motion frame; a one-tick post-clip command-reset mismatch is reported separately.

| model | SR | clustered 95% CI | successes | mean reached frame | object pos RMSE | object ori RMSE | body pos RMSE | pushes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 semantic_uniform | 100.00% | 100.00%–100.00% | 320/320 | 324.0 | 0.0775 m | 5.51° | 0.0462 m | 968 |
| S2 semantic_adaptive | 100.00% | 100.00%–100.00% | 320/320 | 324.0 | 0.0701 m | 5.13° | 0.0445 m | 968 |
| Omni baseline | 100.00% | 100.00%–100.00% | 320/320 | 324.0 | 0.0810 m | 6.12° | 0.0478 m | 966 |

### S1 semantic_uniform

- Success by episode: `[32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- Exclusive failure reasons: `{}`
- Raw failure flags: `{}`
- Timeout/post-reset flags excluded from failure: `{'bad_object_pos': 2, 'bad_object_ori': 2}`
- Failure phases: `{}`
- Survival at frames: `{'50': 1.0, '110': 1.0, '195': 1.0, '270': 1.0, '324': 1.0}`

### S2 semantic_adaptive

- Success by episode: `[32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- Exclusive failure reasons: `{}`
- Raw failure flags: `{}`
- Timeout/post-reset flags excluded from failure: `{'bad_object_pos': 1, 'bad_object_ori': 1}`
- Failure phases: `{}`
- Survival at frames: `{'50': 1.0, '110': 1.0, '195': 1.0, '270': 1.0, '324': 1.0}`

### Omni baseline

- Success by episode: `[32, 32, 32, 32, 32, 32, 32, 32, 32, 32]`
- Exclusive failure reasons: `{}`
- Raw failure flags: `{}`
- Timeout/post-reset flags excluded from failure: `{'bad_object_pos': 1, 'bad_object_ori': 1}`
- Failure phases: `{}`
- Survival at frames: `{'50': 1.0, '110': 1.0, '195': 1.0, '270': 1.0, '324': 1.0}`
