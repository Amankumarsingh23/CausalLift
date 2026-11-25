# Session 12 completion report

Implemented DM, IPS, SNIPS and DR, paired outcome and profit comparisons, ROI, and matching and weight diagnostics. The benchmark reuses earlier models and synthetic allocations, adds released Criteo 200k and 1M evaluations, and records data, model and action fingerprints. See the [evaluation guide](OFFLINE_POLICY_EVALUATION.md) and [business findings](BUSINESS_CASE_STUDY.md).

## Repeated simulation

Each design used 80 replications of 1,600 rows, a fixed covariate rule, known logging propensities, and deliberately imperfect outcome predictions. Bias and RMSE are in percentage points of incremental outcome rate.

| Logging probability | Method | Bias | RMSE | 95% interval coverage |
| ---: | --- | ---: | ---: | ---: |
| 50% | DM | -3.04 | 3.05 | 0% |
| 50% | IPS | -0.03 | 1.45 | 96.25% |
| 50% | DR | -0.10 | 1.27 | 97.50% |
| 15% | DM | -3.02 | 3.03 | 0% |
| 15% | IPS | -0.08 | 2.29 | 96.25% |
| 15% | DR | -0.08 | 1.88 | 96.25% |

Mean effective sample size fell from 801 to 568 as logging became imbalanced, despite the raw match fraction rising from 50% to 64%. Both weighted estimators became noisier.

## Artifacts and handoff

The [full benchmark](../artifacts/session_12/session_12_summary.json) generated 1,296 synthetic and 120 Criteo estimator rows. The offline synthetic fixture also ran successfully. Row-level assignments remain local. The [Session 14 protocol](../artifacts/session_12/final_evaluation_protocol.json) freezes saved models, economics, metrics and test action construction. The reserved test split remains unused. Session 13 can use these validation results and the existing reliability labels while preserving the locked final evaluation for Session 14.
