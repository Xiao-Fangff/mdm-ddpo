# Step experiment analysis

## Executive conclusion

The epsilon MDM does not provide a useful number response by text alone
(counterfactual hard target-count Spearman `0.0959`). Explicit count native
diffusion SFT is the main successful intervention: epoch 2 raises Spearman to
`0.4749`, lowers hard MAE by `0.1701`, and improves exact by `0.0382` while its
held-out HumanML retrieval/M2M deltas remain inside noise.

Shared LoRA+count DDPO improves MAE but eventually makes HumanML validation
infeasible. Count-only DDPO preserves the HumanML no-count path exactly. At
`lr=3e-5` it is too weak; at `lr=1e-4, clip=1e-3`, epoch 5 produces the first
hard acceptance point. This is a promising diagnostic checkpoint, not yet a
production result: exact improves at only one validation point and ankle
high-frequency energy increases slightly.

## Same-pool counterfactual number probe

All rows use identical initial noise, DDIM transition noise, motion length and
prompt template; only target 1--6 changes.

| policy | target Spearman | hard MAE | exact | within-one | ankle HF |
| --- | ---: | ---: | ---: | ---: | ---: |
| original epsilon MDM | 0.095915 | 2.388889 | 0.121528 | 0.364583 | 0.356536 |
| count SFT epoch 2 | 0.474889 | 2.218750 | 0.159722 | 0.399306 | 0.362223 |
| count-only DDPO epoch 5 | 0.504257 | 2.107639 | 0.163194 | 0.427083 | 0.367033 |

The final DDPO policy therefore strengthens the number effect rather than only
changing a global cadence. Relative to SFT it improves counterfactual MAE by
`0.111111`, exact by `0.003472`, within-one by `0.027778`, and Spearman by
`0.029368`. The accompanying ankle-HF increase is `0.004810`.

## Optimizer and preservation audit

For the successful count-only run:

- old/new log-probability audit maximum error: `0`;
- 24/24 finite optimizer updates, zero skipped updates;
- ratio std: `3.64e-5` to `4.32e-5`; clip fraction: `0`;
- count update norm: `0.00782` to `0.01903`;
- LoRA update norm: exactly `0`;
- fixed HumanML retrieval and M2M delta: exactly `0` at every validation.

The fixed step improvements become monotonic in MAE from epoch 0 onward, but
the hard exact delta is flat/negative until epoch 5. Its final `+0.002604`
delta is also smaller than its bootstrap SE (`0.009146`). Reproduction with
seeds 43 and 44 and manual motion inspection remain required before extending
the run or claiming robust count control.

## Fixed step validation

| run | state | epoch | reward delta | exact delta | within-one delta | MAE delta | acceptance points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_count_sft_step_k8_short | best_step | 5 | 0.009355 | -0.005208 | 0.010417 | -0.039062 | 0/6 |
| epsilon_count_sft_step_k8_short | final | 5 | 0.009355 | -0.005208 | 0.010417 | -0.039062 | 0/6 |
| epsilon_count_only_step_k8_short | best_step | 5 | 0.003163 | 0.000000 | 0.002604 | -0.010417 | 0/3 |
| epsilon_count_only_step_k8_short | final | 5 | 0.003163 | 0.000000 | 0.002604 | -0.010417 | 0/3 |
| epsilon_count_only_step_k8_lr1e4_clip1e3 | best_step | 5 | 0.025885 | 0.002604 | 0.033854 | -0.109375 | 1/6 |
| epsilon_count_only_step_k8_lr1e4_clip1e3 | final | 5 | 0.025885 | 0.002604 | 0.033854 | -0.109375 | 1/6 |

## Training advantage statistics (mean over all epochs)

| run | zero variance | step group std median | step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_count_sft_step_k8_short | 0.000000 | 0.209019 | 0.633048 | 0.172055 | 0.171780 | 0.389323 | 0.471354 |
| epsilon_count_only_step_k8_short | 0.000000 | 0.204797 | 0.628236 | 0.171721 | 0.171630 | 0.382812 | 0.466146 |
| epsilon_count_only_step_k8_lr1e4_clip1e3 | 0.000000 | 0.218867 | 0.627392 | 0.174770 | 0.176286 | 0.402344 | 0.462240 |

## Per-target fixed step eval: epsilon_count_sft_step_k8_short

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.906250 | 3.796875 | 3.093750 | 3.046875 | 0.109375 | 0.078125 | 0.343750 | 0.359375 |
| 2 | 3.703125 | 3.625000 | 2.234375 | 2.156250 | 0.046875 | 0.062500 | 0.359375 | 0.406250 |
| 3 | 3.968750 | 3.953125 | 1.843750 | 1.859375 | 0.187500 | 0.203125 | 0.531250 | 0.515625 |
| 4 | 6.046875 | 5.953125 | 2.671875 | 2.671875 | 0.140625 | 0.125000 | 0.312500 | 0.312500 |
| 5 | 8.484375 | 8.406250 | 3.484375 | 3.406250 | 0.062500 | 0.062500 | 0.234375 | 0.250000 |
| 6 | 10.375000 | 10.296875 | 4.718750 | 4.671875 | 0.031250 | 0.015625 | 0.140625 | 0.140625 |

## Per-target fixed step eval: epsilon_count_only_step_k8_short

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.906250 | 3.875000 | 3.093750 | 3.093750 | 0.109375 | 0.093750 | 0.343750 | 0.359375 |
| 2 | 3.703125 | 3.718750 | 2.234375 | 2.218750 | 0.046875 | 0.062500 | 0.359375 | 0.359375 |
| 3 | 3.968750 | 3.968750 | 1.843750 | 1.843750 | 0.187500 | 0.187500 | 0.531250 | 0.546875 |
| 4 | 6.046875 | 6.078125 | 2.671875 | 2.734375 | 0.140625 | 0.140625 | 0.312500 | 0.296875 |
| 5 | 8.484375 | 8.437500 | 3.484375 | 3.437500 | 0.062500 | 0.062500 | 0.234375 | 0.234375 |
| 6 | 10.375000 | 10.312500 | 4.718750 | 4.656250 | 0.031250 | 0.031250 | 0.140625 | 0.140625 |

## Per-target fixed step eval: epsilon_count_only_step_k8_lr1e4_clip1e3

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.906250 | 3.546875 | 3.093750 | 2.796875 | 0.109375 | 0.093750 | 0.343750 | 0.406250 |
| 2 | 3.703125 | 3.609375 | 2.234375 | 2.140625 | 0.046875 | 0.062500 | 0.359375 | 0.406250 |
| 3 | 3.968750 | 3.812500 | 1.843750 | 1.843750 | 0.187500 | 0.203125 | 0.531250 | 0.515625 |
| 4 | 6.046875 | 5.906250 | 2.671875 | 2.625000 | 0.140625 | 0.140625 | 0.312500 | 0.343750 |
| 5 | 8.484375 | 8.343750 | 3.484375 | 3.343750 | 0.062500 | 0.078125 | 0.234375 | 0.281250 |
| 6 | 10.375000 | 10.265625 | 4.718750 | 4.640625 | 0.031250 | 0.015625 | 0.140625 | 0.171875 |
