# K16 step M2M ablation analysis

## Fixed step validation

| run | state | epoch | reward delta | exact delta | within-one delta | MAE delta | acceptance points |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| humanml_step_k16_no_m2m_tb64_fresh | best_step | 84 | 0.004630 | 0.005208 | 0.005208 | -0.014323 | 15/20 |
| humanml_step_k16_no_m2m_tb64_fresh | final | 99 | 0.003616 | 0.003906 | 0.003906 | -0.013021 | 15/20 |
| humanml_step_k16_with_m2m_tb64_fresh | best_step | 79 | 0.004024 | 0.005208 | 0.002604 | -0.011719 | 17/20 |
| humanml_step_k16_with_m2m_tb64_fresh | final | 99 | 0.003287 | 0.002604 | 0.006510 | -0.014323 | 17/20 |

## Training advantage statistics (mean over all epochs)

| run | zero variance | step group std median | step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| humanml_step_k16_no_m2m_tb64_fresh | 0.010000 | 0.173906 | 0.151068 | 0.274020 | 0.201718 | 0.421250 | 0.447500 |
| humanml_step_k16_with_m2m_tb64_fresh | 0.010000 | 0.174291 | 0.151200 | 0.274203 | 0.273761 | 0.417500 | 0.447500 |

## Per-target fixed step eval: humanml_step_k16_no_m2m_tb64_fresh

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.757812 | 2.734375 | 1.757812 | 1.734375 | 0.382812 | 0.382812 | 0.562500 | 0.578125 |
| 2 | 4.000000 | 3.976562 | 2.000000 | 1.976562 | 0.007812 | 0.007812 | 0.179688 | 0.187500 |
| 3 | 3.804688 | 3.796875 | 0.882812 | 0.875000 | 0.250000 | 0.257812 | 0.882812 | 0.882812 |
| 4 | 4.242188 | 4.234375 | 0.429688 | 0.421875 | 0.648438 | 0.656250 | 0.921875 | 0.921875 |
| 5 | 4.835938 | 4.843750 | 0.914062 | 0.906250 | 0.273438 | 0.281250 | 0.851562 | 0.851562 |
| 6 | 6.171875 | 6.164062 | 1.718750 | 1.710938 | 0.242188 | 0.242188 | 0.515625 | 0.515625 |

## Per-target fixed step eval: humanml_step_k16_with_m2m_tb64_fresh

| target | baseline detected | final detected | baseline MAE | final MAE | baseline exact | final exact | baseline within-one | final within-one |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.757812 | 2.726562 | 1.757812 | 1.726562 | 0.382812 | 0.382812 | 0.562500 | 0.578125 |
| 2 | 4.000000 | 3.960938 | 2.000000 | 1.960938 | 0.007812 | 0.007812 | 0.179688 | 0.203125 |
| 3 | 3.804688 | 3.789062 | 0.882812 | 0.867188 | 0.250000 | 0.265625 | 0.882812 | 0.882812 |
| 4 | 4.242188 | 4.226562 | 0.429688 | 0.429688 | 0.648438 | 0.648438 | 0.921875 | 0.921875 |
| 5 | 4.835938 | 4.828125 | 0.914062 | 0.921875 | 0.273438 | 0.273438 | 0.851562 | 0.851562 |
| 6 | 6.171875 | 6.164062 | 1.718750 | 1.710938 | 0.242188 | 0.242188 | 0.515625 | 0.515625 |
