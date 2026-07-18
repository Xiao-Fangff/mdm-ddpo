# K16 step DDPO diagnosis and next experiment

## Existing runs

The full metric and target tables are in
`reports/k16_step_m2m_ablation_analysis.md` and its JSON companion.

- Both runs completed 100 epochs and 20 fixed validation points.
- No-step-M2M improved all three acceptance directions at 15/20 validation
  points; with-step-M2M did so at 17/20 points.
- The best fixed deltas are nevertheless extremely small:
  - no-step-M2M: reward `+0.004630`, exact `+0.005208`, within-one
    `+0.005208`, MAE `-0.014323`;
  - with-step-M2M: reward `+0.004024`, exact `+0.005208`, within-one
    `+0.002604`, MAE `-0.011719`.

This is a persistent but very weak fixed-eval effect, rather than a single
rollout spike.

## Advantage diagnosis

Mean values over all 100 epochs:

| run | zero variance | group std median | hard-step contribution | retrieval contribution | M2M contribution | retrieval-step conflict | M2M-step conflict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no step M2M | 0.0100 | 0.1739 | 0.1511 | 0.2740 | 0.2017 (HumanML only) | 0.4213 | 0.4475 |
| with step M2M | 0.0100 | 0.1743 | 0.1512 | 0.2742 | 0.2738 | 0.4175 | 0.4475 |

- K16 already reduced zero-variance groups to about 1%, so zero variance is
  not the current bottleneck.
- Hard-step contribution is only about 55% of retrieval contribution.
- With step M2M enabled, its step-sample contribution is `0.2878`, larger than
  the hard-step contribution `0.1512`, while M2M conflicts with the count
  direction about 45% of the time.
- The current global `.375/.375/.25` weights therefore do not isolate whether
  count itself is learnable.

## Fixed target diagnosis

The final detected means are not all collapsed to one constant, but the model
barely moves them from baseline. For the no-step-M2M run:

| target | baseline detected | final detected | baseline MAE | final MAE | final exact |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.7578 | 2.7344 | 1.7578 | 1.7344 | 0.3828 |
| 2 | 4.0000 | 3.9766 | 2.0000 | 1.9766 | 0.0078 |
| 3 | 3.8047 | 3.7969 | 0.8828 | 0.8750 | 0.2578 |
| 4 | 4.2422 | 4.2344 | 0.4297 | 0.4219 | 0.6563 |
| 5 | 4.8359 | 4.8438 | 0.9141 | 0.9063 | 0.2813 |
| 6 | 6.1719 | 6.1641 | 1.7188 | 1.7109 | 0.2422 |

Target 2 is the clearest failure: final exact accuracy is below 1%. The small
changes after 100 epochs indicate that two independent step prompts per epoch
do not provide enough text-number diversity.

## Detector GT/reference audit

`tools/validate_step_detector_gt.py` was run on all 1,842 reference XYZ motions
whose original captions requested targets 1–6. The local outputs are
`artifacts/step_gt_detector_validation.json/.md`.

- Original target 1 has no samples in this manifest.
- Overall requested-target vs detected exact accuracy: `0.1401`.
- Overall MAE: `2.6118`; within-one accuracy: `0.3621`.
- Target 6: detected mean `2.9604`, exact `0.1189`, MAE `3.0396`.
- Re-detection reproduces the manifest's stored pseudo label with accuracy
  `1.0`.

Therefore the manifest labels are internally consistent detector pseudo labels,
but they are not reliable annotations of the original caption's requested step
count. The next K8 experiment is an isolation test of pseudo-count learnability;
it must not be interpreted as validated real-number control.

## Next diagnostic configuration

- HumanML groups: `0.5 retrieval + 0.5 M2M`.
- Step groups: `0.2 retrieval + 0.0 M2M + 0.8 step`.
- `step_data_ratio=0.5`, `K_step=8`, four step prompts per rollout and 16 per
  epoch.
- Balanced target sampler over pseudo targets 1–6.
- `negative_l1` hard reward and a matching K8 calibration.
- Select and inspect `best_step.pt`; HumanML `best_balanced.pt` remains a
  separate preservation criterion.

Acceptance requires multiple fixed validation points with MAE delta below zero,
exact delta above zero, and within-one delta above zero.
