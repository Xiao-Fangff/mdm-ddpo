# DDPO–MDM timestep and fixed-trajectory diagnostic

Date: 2026-07-19

The complete machine-readable result is
`reports/ddpo_mdm_compatibility_diagnostic.json`.

## Setup

- Original 50-step cosine, x-start-prediction MDM with zero-initialized LoRA.
- Batch 16: 8 HumanML motions and one K8 synthetic step prompt group.
- Step target in the deterministic seed-42 rollout: 2.
- All 49 stochastic transitions, learning rate `1e-4`, clip range `1e-3`.
- Six fixed-trajectory optimizer updates.
- HumanML advantages were zeroed during the overfit updates so the test isolates
  step-count credit assignment.
- Initial latent, every transition noise, prompt, length, and target were replayed
  exactly after every update.

## Correctness checks

- Full-trajectory old/new log-probability audit max error: `0`.
- Recovered transition-noise reconstruction max error: `4.77e-7`.
- Initial fixed-noise reward replay max error: `8.48e-5`.
- No skipped or non-finite optimizer update.

These checks rule out trajectory/action misalignment and a reversed or disconnected
score-function gradient in this diagnostic.

## Timestep imbalance

| timestep bucket | theoretical x0 score sensitivity mean | empirical advantage-gradient norm | post-update clip fraction |
| --- | ---: | ---: | ---: |
| 1–2 | 14.6853 | 0.017852 | 0.9063 |
| 3–5 | 4.5058 | 0.005903 | 0.1250 |
| 6–15 | 1.3496 | 0.000973 | 0.0000 |
| 16–30 | 0.4092 | 0.000526 | 0.0000 |
| 31–49 | 0.1428 | 0.000181 | 0.0000 |

Across individual timesteps, theoretical x-start score sensitivity ranges from
`0.03116` to `19.69975`, a `632.14x` ratio. The empirical gradient norm of the
`t=1–2` bucket is about `98.6x` the `t=31–49` bucket even though their mean
absolute advantages are comparable.

After only one update, 90.6% of `t=1–2` actions would be outside a `1e-3` PPO
clip range, while every transition at `t>=6` remains inside it. After six updates,
the cumulative clip fractions are 100% for `t=1–2`, 68.8% for `t=3–5`, 2.5% for
`t=6–15`, and zero for `t>=16`.

This confirms that aggregate clip fraction hides severe concentration in the final
few reverse-diffusion transitions.

## Fixed-trajectory overfit

| metric | update 0 | update 6 | change |
| --- | ---: | ---: | ---: |
| step reward | -0.531360 | -0.527354 | +0.004005 |
| soft count, target 2 | 4.168209 | 4.158542 | -0.009667 |
| hard count mean | 4.25 | 4.25 | 0 |
| hard MAE | 2.25 | 2.25 | 0 |
| effective LoRA delta norm | 0 | 0.157368 | +0.157368 |

The cumulative advantage/log-probability correlation reaches `0.5672`; the mean
log-probability delta gap between positive- and negative-advantage actions is
`+0.001118`, and the advantage-weighted log-probability delta is positive
(`+0.000225`). Therefore DDPO assigns credit in the correct relative direction.

Nevertheless, even deliberately overfitting the same target-2 trajectory moves the
continuous count by only `0.0097` and never changes the hard count. The bottleneck is
not a missing gradient; it is the combination of highly concentrated timestep credit
and weak motion-level response.

## Hypothetical epsilon parameterization

Under the same schedule, an epsilon-prediction transition would have sensitivity
range `0.36447` to `31.62232`, or `86.76x`. Its bucket means are:

| timestep bucket | epsilon score sensitivity mean |
| --- | ---: |
| 1–2 | 1.2385 |
| 3–5 | 0.7333 |
| 6–15 | 0.4739 |
| 16–30 | 0.3725 |
| 31–49 | 2.2518 |

Epsilon prediction is substantially better balanced across most timesteps and gives
more credit to the high-noise beginning of reverse diffusion. It still has a large
`t=49` endpoint spike, so changing prediction type alone does not remove the need for
per-timestep diagnostics or normalization.

## Conclusion

The DDIM transition and PPO credit direction are functioning. The diagnostic confirms
a real x-start/DDPO optimization mismatch: almost all effective update and clipping is
concentrated at `t<=5`, while high-noise transitions that can establish global count
structure receive much weaker gradients.

Training an epsilon-prediction MDM is therefore justified as a controlled ablation,
but not yet as a guaranteed replacement. It must use the same architecture, data,
schedule, baseline evaluation, fixed pools, and optimizer-update budget. The epsilon
run should also retain per-timestep logging because its final high-noise endpoint can
become the new clipping bottleneck.
