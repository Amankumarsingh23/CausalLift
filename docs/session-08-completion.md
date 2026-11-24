# Session 08 completion

The run produced DAG identification checks, numerical bias examples, three-fold cross-fitted assignment scores, calibration, empirical overlap and unadjusted balance. It used fixed train and validation partitions, saved and reloaded 12 final propensity models, and left test reserved. The default run took 8.55 seconds on Python 3.13.15 with NetworkX 3.7.

## Graphs and simulations

| Graph | Empty set | Proposed conditioning | True total effect | Unadjusted | Conditioned |
| --- | --- | --- | ---: | ---: | ---: |
| Observed confounding | Fails | X identifies | 1 | 2.900 | 0.987 |
| Mediation | Identifies | M invalid for total effect | 4 | 4.006 | 1.015 |
| Collider | Identifies | C opens biasing path | 1 | 1.016 | 0.020 |
| Hidden confounding | Fails | U unavailable | — | — | — |

The collider path demonstration uses a graph with only the direct T → Y edge removed so that the opened path is isolated. The total-effect checker separately rejects C as a treatment descendant. [Graphs](../artifacts/session_08/causal_graphs.png) · [Identification JSON](../artifacts/session_08/identification_results.json)

## Propensity and overlap

| Cohort | Logistic AUC / Brier | LightGBM AUC / Brier | Observed share | Logistic / LightGBM predicted means |
| --- | --- | --- | ---: | --- |
| A | .497 / .2502 | .478 / .2527 | .4959 | .4957 / .4943 |
| C | .844 / .1584 | .844 / .1586 | .4403 | .4395 / .4400 |
| F | .827 / .1699 | .825 / .1710 | .4999 | .4919 / .4910 |
| Criteo 200k | .509 / .1275 | .508 / .1275 | .8500 | .8499 / .8499 |

The constant score had AUC 0.5 in each cohort. On C, propensity RMSE was 0.0098 for logistic, 0.0404 for LightGBM and 0.2975 for constant. Logistic placed 2.15% outside empirical support and 10.45% in the tails. Full-information F scores placed 12.60% outside support and 45.31% in the tails, versus 4.67% and 6.12% for F logistic. F scores condition on different information, so these differences are not pure estimation error. No rows were trimmed. Criteo's original probabilities are unavailable. [Model metrics](../artifacts/session_08/propensity_model_results.csv) · [Calibration](../artifacts/session_08/propensity_calibration.csv) · [Overlap](../artifacts/session_08/propensity_overlap_summary.csv)

At the absolute SMD threshold of 0.1, A flagged no features. C flagged engagement (1.132) and loyalty (0.667), and F flagged engagement (1.335). [Balance](../artifacts/session_08/covariate_balance.csv)

## Session 07 follow-up

The three original 2,400-row draws reproduced DML estimates of 1.499, 1.552 and 1.504 for a true effect of 1.25. Outcome OOF RMSE was 1.50–1.58 versus a structural noise SD of 1. Logistic propensity RMSE against design was .194–.204, compared with .138–.148 for LightGBM. Treatment residual SD was .500–.503 and .479–.485 respectively. The diagnostic includes score calibration and overlap by draw. Finite-sample nuisance errors and nonlinear misspecification are plausible contributors. This comparison cannot isolate a cause or change the original 0/30 coverage finding. [Nuisance diagnostics](../artifacts/session_08/nuisance_quality_diagnostics.json)

## Reproduction and handoff

Run `python -m causallift.session08` or use `--fixture` for offline verification. Run `python -m pytest -q`, `python -m ruff check .` and `python -m pip check` for checks. The [manifest](../artifacts/session_08/propensity_predictions_manifest.json) documents local-only Joblib and Parquet artifacts, OOF folds and row provenance. Session 09 can reuse the scores after verifying the partition fingerprints and evaluating identification and support for its target population. [DAG guide](CAUSAL_DAGS.md) · [Propensity guide](PROPENSITY_SCORES.md)
