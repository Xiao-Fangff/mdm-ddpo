# Epsilon linear-50 Min-SNR-5 MDM evaluation

Date: 2026-07-21

## Checkpoint and effective configuration

- Checkpoint: `/home/zhiwei/projects/motion-diffusion-model/save/humanml_trans_dec_512_bert_epsilon_linear50_minsnr5_xstart1_vel01/model000150000.pt`
- Prediction type: `epsilon`
- Training and DDPO sample steps: `50`
- Noise schedule: `linear`
- Variance: `FIXED_SMALL` (`sigma_small=true`)
- Pretraining losses: Min-SNR gamma `5`, x-start MSE auxiliary `1.0`,
  x-start velocity auxiliary `0.1`
- DDPO sampler: stochastic DDIM, `eta=1.0`, guidance `2.5`

The local runtime read these values from the paired `args.json` and audited the
effective diffusion object.  No external repository was modified.

This checkpoint is not a prediction-type-only ablation against the original x-start
MDM: its schedule and pretraining losses also differ.

## Base model quality

The supplied five-replication HumanML debug evaluation reports matching score
`3.8177 +/- 0.0446`, R-precision top-1/2/3 `0.3770/0.5570/0.6736`, FID
`1.3579 +/- 0.1139`, and diversity `8.7665 +/- 0.1193`.

On the shared 1024-prompt reward calibration pool, the epsilon checkpoint has lower
absolute MotionReward means than the original x-start checkpoint, but substantially
more within-prompt variation and better retrieval/M2M agreement:

| metric | x-start | epsilon linear-50 |
| --- | ---: | ---: |
| retrieval mean | 0.7889 | 0.5618 |
| M2M mean | 0.7960 | 0.5124 |
| retrieval within-group std p25 | 0.0115 | 0.0733 |
| M2M within-group std p25 | 0.0117 | 0.0793 |
| within-group correlation | 0.5851 | 0.8154 |
| ranking conflict fraction | 0.3644 | 0.1955 |

## DDPO compatibility diagnostic

At learning rate `3e-5` and PPO clip `1e-3`:

- full initial old/new log-probability max difference: `0`;
- exact transition-noise replay max error: `4.77e-7`;
- effective epsilon score-sensitivity max/min ratio: `4.68x`;
- initial empirical timestep-bucket gradient max/min ratio: `9.81x`;
- six-update fixed step reward: `0.01771 -> 0.02020`;
- advantage/log-probability delta correlation after six updates: `0.963`;
- skipped or non-finite updates: `0`.

The parameterization therefore gives usable policy gradients across all timesteps,
although `t=1--2` still has the largest empirical gradient.

## Step-count baseline

The shared synthetic K8 calibration has rich continuous ranking signal (mean
`7.94/8` unique reward levels, `0.27%` pairwise ties, zero zero-variance groups),
but poor hard-count quality:

| metric | x-start | epsilon linear-50 |
| --- | ---: | ---: |
| hard MAE | 1.3268 | 2.7204 |
| hard exact | 0.2493 | 0.1139 |
| hard within-one | 0.6143 | 0.3561 |
| soft-hard absolute difference | 0.2147 | 0.6391 |
| ankle high-frequency ratio | 0.00894 | 0.32470 |

In the stricter counterfactual pool (same noise, length, and template; only the
number changes), hard target-count Spearman is `0.0959`, while length-count Spearman
is `0.4271`.  Detected means for targets 1--6 remain concentrated between `3.52`
and `4.40`.  The text embeddings are numerically separable, but generated step count
barely uses the number condition.

## Six-epoch isolated DDPO result

The run used K8, 50% step motions, 16 independent step prompt groups per epoch,
step-only soft advantage on step samples, and HumanML retrieval/M2M protection.
PPO stayed stable: aggregate clip fraction was at most `0.44%`, no update was
skipped, and the initial ratio audit was exact.

| state | soft reward delta | soft MAE delta | hard MAE delta | exact delta | within-one delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| best soft epoch 4 | +0.00405 | -0.01532 | -0.01302 | -0.00781 | -0.00521 |
| final epoch 5 | +0.00353 | -0.01587 | -0.00260 | -0.01042 | -0.00260 |

There were `0/6` hard acceptance points.  At epoch 4 the soft reward and soft MAE
changes were only about `1.3` bootstrap standard errors.  HumanML retrieval/M2M
remained within their paired standard errors and were feasible at five of six
validation points.

The final counterfactual probe changed hard MAE by `-0.0382`, but target-count
Spearman changed from `0.0959` to only `0.0932`; length-count Spearman remained
`0.4247`, and ankle high-frequency ratio increased by `0.00565`.  This is a mostly
global downward count shift, not learned numeric control.

## Decision

Do not extend this configuration directly to 30 or 100 epochs.  Epsilon prediction
substantially repairs DDPO timestep credit, but this checkpoint still does not use
number conditioning strongly, and its motions expose a large detector/jitter
mismatch.  The next justified step is explicit count conditioning plus a short
native diffusion SFT (with an anti-jitter/quality gate), followed by the same
counterfactual probe before more DDPO.

Machine-readable artifacts:

- `reports/ddpo_mdm_epsilon_linear50_minsnr5_150k_diagnostic.json`
- `reports/epsilon_linear50_minsnr5_150k_counterfactual_number_probe.json`
- `reward_calibration_epsilon_linear50_minsnr5_150k.json`
- `step_reward_k8_soft_huber_epsilon_linear50_minsnr5_150k.json`
- `reports/epsilon_linear50_minsnr5_150k_step_k8_short_analysis.json`
- `reports/epsilon_linear50_minsnr5_150k_step_k8_short_counterfactual.json`
