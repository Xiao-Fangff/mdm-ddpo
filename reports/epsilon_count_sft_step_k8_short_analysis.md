# Step experiment analysis

## Fixed step validation

| run | state | epoch | reward delta | exact delta | within-one delta | MAE delta | acceptance points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_count_sft_step_k8_short | best_step | 5 | 0.009355 | -0.005208 | 0.010417 | -0.039062 | 0/6 |
| epsilon_count_sft_step_k8_short | final | 5 | 0.009355 | -0.005208 | 0.010417 | -0.039062 | 0/6 |

## Training advantage statistics (mean over all epochs)

| run | zero variance | step group std median | step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| epsilon_count_sft_step_k8_short | 0.000000 | 0.209019 | 0.633048 | 0.172055 | 0.171780 | 0.389323 | 0.471354 |

## Per-target fixed step eval: epsilon_count_sft_step_k8_short

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.906250 | 3.796875 | 3.093750 | 3.046875 | 0.109375 | 0.078125 | 0.343750 | 0.359375 |
| 2 | 3.703125 | 3.625000 | 2.234375 | 2.156250 | 0.046875 | 0.062500 | 0.359375 | 0.406250 |
| 3 | 3.968750 | 3.953125 | 1.843750 | 1.859375 | 0.187500 | 0.203125 | 0.531250 | 0.515625 |
| 4 | 6.046875 | 5.953125 | 2.671875 | 2.671875 | 0.140625 | 0.125000 | 0.312500 | 0.312500 |
| 5 | 8.484375 | 8.406250 | 3.484375 | 3.406250 | 0.062500 | 0.062500 | 0.234375 | 0.250000 |
| 6 | 10.375000 | 10.296875 | 4.718750 | 4.671875 | 0.031250 | 0.015625 | 0.140625 | 0.140625 |
