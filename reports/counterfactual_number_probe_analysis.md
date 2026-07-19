# Counterfactual number-conditioning probe

Pool ID: `52127be993e53464a44b96d2524c4e382d3765b1eacbc5539f0b640b77e0bf69`

The pool crosses four shared-support motion lengths with six identical prompt
templates.  For every length/template/noise condition, targets 1--6 reuse the
same initial noise and every DDIM transition noise.  The probe uses 24
conditions, two noise samples per condition, six targets, and 50 diffusion
steps (288 generated motions per policy).

## Original MDM

| metric | hard count | soft count (tau=0.25) | soft count (tau=1.0) |
| --- | ---: | ---: | ---: |
| target-count Spearman | 0.4540 | 0.4965 | 0.4915 |
| target regression coefficient | 0.3304 | 0.3300 | 0.3322 |
| target standardized coefficient | 0.5247 | 0.5243 | 0.5483 |
| length standardized coefficient | 0.1810 | 0.1818 | 0.1093 |
| mean absolute soft-hard difference | -- | 0.0051 | 0.1742 |

Projected `embed_text` outputs are numerically distinct when only the number
word changes: minimum pairwise RMS distance is `0.02306`, and minimum cosine
distance is `0.02025`.  Adjacent-number prompts also change generated motions
(mean normalized-motion RMS distance `0.3148`).

Therefore both the text representation and the diffusion policy use number
tokens.  The explicit count-embedding fallback is not currently justified.
The problem is weak/nonlinear count control, not total absence of numeric
conditioning.

Original hard-count target means are:

| target | detected mean | MAE | exact | within-one |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.0625 | 1.0625 | 0.5625 | 0.6250 |
| 2 | 3.8125 | 1.8125 | 0.0000 | 0.2708 |
| 3 | 3.8542 | 0.8542 | 0.2083 | 0.9375 |
| 4 | 3.8542 | 0.3125 | 0.6875 | 1.0000 |
| 5 | 4.2292 | 0.9375 | 0.1875 | 0.8750 |
| 6 | 4.1250 | 1.8750 | 0.0417 | 0.2292 |

The response is target-dependent but saturates around four steps.  Target 2
is not merely a length artifact: its exact accuracy is zero even in the fully
crossed counterfactual pool.

## Existing negative-L1 K8 checkpoints

| policy | hard MAE | exact | within-one | hard target Spearman | soft target Spearman |
| --- | ---: | ---: | ---: | ---: | ---: |
| original MDM | 1.14236 | 0.28125 | 0.65625 | 0.45403 | 0.49652 |
| epoch 11 | 1.14583 | 0.28125 | 0.65278 | 0.44934 | 0.48923 |
| epoch 29 | 1.13542 | 0.28472 | 0.65625 | 0.46389 | 0.50999 |

Epoch 29 makes only a very small counterfactual improvement: MAE `-0.00694`,
exact `+0.00347`, hard correlation `+0.00986`.  Epoch 11 slightly regresses on
this deconfounded pool despite looking better on the old pseudo-reference
fixed pool.  This confirms that the previous checkpoint selection was partly
specific to the confounded validation pool.

## Soft detector temperature

With normalized-margin temperatures of `0.25`, kept candidate probabilities
are almost saturated and the mean soft-hard difference is only `0.0051`.
Increasing lead/length/progress temperatures to `1.0` raises this difference
to `0.1742` while preserving target correlation (`0.4915`).  The default is
therefore `1.0`; every production run must generate a new matching calibration
before training.

Ankle high-frequency ratios remain essentially unchanged for the old run
(`0.00901` original versus `0.00903` at epoch 29), so the existing weak gain is
not explained by visible high-frequency detector exploitation.
