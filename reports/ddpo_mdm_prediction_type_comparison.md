# DDPO–MDM prediction-type comparison

Date: 2026-07-19

Machine-readable reports:

- `reports/ddpo_mdm_compatibility_diagnostic.json`: 50-step x-start, six updates.
- `reports/ddpo_mdm_epsilon_compatibility_diagnostic.json`: 50-step epsilon, six updates.
- `reports/ddpo_mdm_epsilon_200step_screen.json`: 200-step epsilon respaced to 50, one-update screen.
- `reports/ddpo_mdm_epsilon_1000step_screen.json`: 1000-step epsilon respaced to 50, one-update screen.

## Loader correctness

The epsilon checkpoint `args.json` contains `"predict_epsilon": true`, while the
external MDM checkout currently constructs `START_X` unconditionally.  The local
DDPO runtime now resolves `--prediction-type auto` from checkpoint metadata and
overrides the returned diffusion object without modifying the external repository.
Both diagnostic logs explicitly confirmed their effective types (`x_start` and
`epsilon`).

For both six-update runs:

- initial full-trajectory log-probability audit error: `0`;
- transition reconstruction max error: `4.77e-7`;
- reward replay max error: `8.48e-5` (x-start), `1.19e-4` (epsilon);
- no non-finite or skipped optimizer update.

## Timestep credit

The following uses the same seed, 16 samples, 49 stochastic transitions, learning
rate `1e-4`, PPO clip `1e-3`, and 32 sampled sample/timestep pairs per bucket.

| timestep bucket | x-start gradient norm | epsilon gradient norm | x-start first-update clip | epsilon first-update clip |
| --- | ---: | ---: | ---: | ---: |
| 1–2 | 0.017852 | 0.048436 | 90.63% | 31.25% |
| 3–5 | 0.005903 | 0.025606 | 12.50% | 31.25% |
| 6–15 | 0.000973 | 0.015165 | 0% | 25.63% |
| 16–30 | 0.000526 | 0.054797 | 0% | 22.92% |
| 31–49 | 0.000181 | 0.055496 | 0% | 20.72% |

The empirical maximum/minimum bucket-gradient ratio falls from about `98.6x` to
`3.66x`.  The schedule-only raw-output score sensitivity ratio falls from `632.14x`
for x-start to `86.76x` for epsilon.  Epsilon therefore fixes the observed
near-zero early/high-noise credit, although its movement is already too large for a
`1e-3` clip in roughly 20–31% of every bucket.  It still needs a smaller learning
rate, timestep normalization, or an endpoint-aware sampler rather than blindly
reusing the x-start optimizer settings.

## Fixed-trajectory overfit

Only the K8 target-2 step group contributed advantages.  Initial latent, every
transition noise, text, and length were replayed exactly after each optimizer update.

| metric | x-start update 0 | x-start update 6 | epsilon update 0 | epsilon update 6 |
| --- | ---: | ---: | ---: | ---: |
| step reward | -0.531360 | -0.527354 | -0.285001 | -0.170883 |
| soft count | 4.168209 | 4.158542 | 0.500143 | 0.885335 |
| hard count mean | 4.25 | 4.25 | 0.625 | 0.25 |
| hard MAE | 2.25 | 2.25 | 1.625 | 1.75 |
| hard exact | 0 | 0 | 0.125 | 0.125 |
| hard within-one | 0.125 | 0.125 | 0.25 | 0.125 |

Epsilon produces a much larger positive advantage-weighted log-probability movement
(`0.01216` versus `0.000225`) and a larger soft reward response.  It does **not**
improve the hard count: MAE and within-one get worse.  This is evidence that the
soft shaping/detector alignment remains an independent problem; it is not evidence
that the current epsilon checkpoint has learned valid step control.

## Base-model quality gate

All values below are zero-LoRA metrics from the same deterministic diagnostic batch.

| checkpoint | retrieval | M2M | step hard MAE (target 2) |
| --- | ---: | ---: | ---: |
| 50-step x-start | 0.6412 | 0.3214 | 2.250 |
| 50-step epsilon | -0.0824 | 0.0507 | 1.625 |
| 200-step epsilon, sampled in 50 steps | 0.1091 | 0.1688 | 1.500 |
| 1000-step epsilon, sampled in 50 steps | 0.0599 | 0.1312 | 1.375 |

The epsilon baselines are not comparable to the x-start MDM.  A formal
prediction-type ablation must therefore train or select an epsilon base model that
first passes standard HumanML evaluation and the shared fixed-pool baseline.  Only
then should both bases be calibrated independently and receive equal DDPO rollout
and optimizer-update budgets.

## Decision

Prediction type is a confirmed cause of timestep credit imbalance, but it is not the
sole cause of the flat reward curves.  Do not launch a long DDPO run from the current
epsilon checkpoints.  First repair/retrain the epsilon base and require baseline
parity; then retain per-timestep logging because epsilon has a high-noise endpoint
spike and can clip across all buckets.
