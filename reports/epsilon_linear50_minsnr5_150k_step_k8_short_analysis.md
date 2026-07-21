# Step experiment analysis

## Fixed step validation

| run | state | epoch | reward delta | exact delta | within-one delta | MAE delta | acceptance points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_linear50_minsnr5_150k_step_k8_short | best_step | 4 | 0.004048 | -0.007812 | -0.005208 | -0.013021 | 0/6 |
| epsilon_linear50_minsnr5_150k_step_k8_short | final | 5 | 0.003533 | -0.010417 | -0.002604 | -0.002604 | 0/6 |

## Training advantage statistics (mean over all epochs)

| run | zero variance | step group std median | step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_linear50_minsnr5_150k_step_k8_short | 0.000000 | 0.263540 | 0.638019 | 0.176999 | 0.178856 | 0.380208 | 0.453125 |

## Per-target fixed step eval: epsilon_linear50_minsnr5_150k_step_k8_short

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.203125 | 4.171875 | 3.328125 | 3.328125 | 0.093750 | 0.078125 | 0.343750 | 0.343750 |
| 2 | 4.781250 | 4.796875 | 3.031250 | 3.046875 | 0.062500 | 0.062500 | 0.187500 | 0.187500 |
| 3 | 3.828125 | 3.796875 | 1.859375 | 1.890625 | 0.187500 | 0.187500 | 0.500000 | 0.468750 |
| 4 | 4.875000 | 4.843750 | 2.343750 | 2.343750 | 0.187500 | 0.156250 | 0.390625 | 0.406250 |
| 5 | 6.765625 | 6.718750 | 2.421875 | 2.375000 | 0.234375 | 0.250000 | 0.453125 | 0.453125 |
| 6 | 7.609375 | 7.531250 | 3.109375 | 3.093750 | 0.093750 | 0.062500 | 0.296875 | 0.296875 |
