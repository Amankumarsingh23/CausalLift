# CausalLift Session 07 completion report

Partially linear DML, EconML LinearDML and honest CausalForestDML ran on both Criteo cohorts and randomized synthetic A, B, D and E. The combined table covers all six causal learner families. The benchmark reloaded prior predictive and S/T/X/R artifacts, reproduced all 92 shared ranking rows and verified matching treatment capacities and common full-population gain endpoints. Final test partitions stayed reserved. Truth entered only evaluation.

## Environment

EconML 0.17.1 installed through `.[dev,advanced]` into the existing Python 3.13.15 Windows virtual environment. NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1 and LightGBM 4.7.0 stayed in place. The new transitive packages were llvmlite 0.50.0, numba 0.68.1, SHAP 0.54.1, slicer 0.0.8, sparse 0.19.3 and tqdm 4.70.1. Python Install Manager auto-updated to 26.4 during inspection. The Python runtime stayed 3.13.15. Prior-model reload worked without prediction regeneration, and all 16 new Joblib artifacts reproduced their effects in this environment. [EconML package metadata](https://pypi.org/project/econml/0.17.1/)

## Partially linear DML

Results below are in percentage points. Truth is the synthetic training population’s mean conditional effect. Known constant synthetic propensities support the ATE interpretation. Criteo uses the released training-share approximation, with an independent-row standard error and unavailable original trial design details.

| Cohort | Outcome | True training mean CATE | Estimate | SE | 95% CI |
| --- | --- | ---: | ---: | ---: | --- |
| Criteo 200k | conversion | — | 0.096 | 0.036 | [0.025, 0.167] |
| Criteo 200k | visit | — | 0.745 | 0.137 | [0.477, 1.013] |
| synthetic_A | outcome | 10.000 | 10.278 | 0.871 | [8.572, 11.984] |
| synthetic_B | outcome | -0.110 | 0.485 | 0.837 | [-1.155, 2.125] |
| synthetic_D | outcome | -0.043 | -0.932 | 1.151 | [-3.187, 1.324] |
| synthetic_E | outcome | 1.832 | 2.635 | 0.852 | [0.965, 4.305] |
| Criteo 1M | conversion | — | 0.081 | 0.016 | [0.050, 0.113] |
| Criteo 1M | visit | — | 0.678 | 0.060 | [0.560, 0.796] |

## High-dimensional confounding

The true continuous structural effect is 1.25. The simulator used 30 observed covariates, and its realized propensity range across 30 repetitions was approximately 0.100–0.900.

| Method | Mean estimate | Bias | RMSE | 95% intervals covering truth |
| --- | ---: | ---: | ---: | ---: |
| Naive difference | 2.508 | 1.258 | 1.259 | 0/30 |
| Linear adjustment | 2.371 | 1.121 | 1.124 | 0/30 |
| Partially linear DML | 1.570 | 0.320 | 0.325 | 0/30 |

Cross-fitted DML reduced point-estimation error under the fixed simulation design. It retained substantial positive bias and none of its intervals covered truth. These results demonstrate finite-sample nuisance and inference limitations. A small simulation does not establish a general coverage rate. [DML construction and prespecified design](DOUBLE_MACHINE_LEARNING.md)

## Synthetic CATE accuracy

Root PEHE and signed ATE error are in percentage points.

| Scenario | Estimator | Root PEHE | Signed ATE error | Spearman |
| --- | --- | ---: | ---: | ---: |
| A | LinearDML | 1.500 | 0.282 | — |
| A | Causal Forest | 2.028 | -0.028 | — |
| B | LinearDML | 4.963 | 0.498 | 0.881 |
| B | Causal Forest | 4.102 | 0.307 | 0.896 |
| D | LinearDML | 6.603 | -1.250 | 0.756 |
| D | Causal Forest | 5.185 | -1.060 | 0.843 |
| E | LinearDML | 8.925 | 1.036 | 0.791 |
| E | Causal Forest | 3.006 | 0.876 | 0.815 |

The constant baseline still led homogeneous A (0.38 pp). The forest improved on the best earlier learner’s PEHE in B (4.10 versus 4.77 pp), D (5.18 versus 5.99 pp) and E (3.01 versus 5.48 pp). Synthetic forest pointwise coverage and mean interval width are shown below. These are cohort-level summaries of pointwise intervals from single fits, rather than repeated-fit or simultaneous coverage.

| Scenario | Supported intervals | Empirical coverage | Mean width (pp) |
| --- | ---: | ---: | ---: |
| A | 4001 | 99.58% | 13.77 |
| B | 4000 | 86.98% | 14.03 |
| D | 4000 | 90.42% | 19.23 |
| E | 4001 | 96.28% | 13.88 |

Criteo conversion had 52 unsupported forest intervals out of 4,000 sampled rows on 200k and 48 on 1M. EconML returned nonfinite standard errors for those rows. Its confidence interval helper can otherwise present the point estimate as both limits, so the report explicitly checks standard errors. The 1M conversion average-effect interval was [−1.018, +1.217] pp on the 3,952 supported inference rows, using the library’s aggregation. The forest’s 1M visit interval was [−3.263, +4.529] pp on 4,000 rows. The ranking population remained the full validation partition. No advanced validation predictions exceeded [−1, 1]. [Forest inference and limits](CAUSAL_FOREST.md)

## Criteo 1M validation

| Outcome | Estimator | Qini | AUUC | Uplift@10% (pp) | @20% (pp) | @50% (pp) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| conversion | LinearDML | 0.0004156 | 0.0011324 | 1.112 | 0.541 | 0.235 |
| conversion | Causal Forest | 0.0003573 | 0.0010452 | 0.949 | 0.525 | 0.219 |
| visit | LinearDML | 0.0035925 | 0.0114229 | 9.967 | 5.538 | 2.413 |
| visit | Causal Forest | 0.0032553 | 0.0105646 | 8.994 | 5.096 | 2.210 |

Earlier linear and S-LightGBM baselines remained competitive. Saved X-linear conversion Qini was 0.0004171 and S-LightGBM visit Qini was 0.0038515. The 1M forest used a predefined stratified 120,000-row training cap, while LinearDML and partially linear DML used all 600,000 training rows. The common validation population contained 583 conversions, including 58 controls. Qini and AUUC retain the Session 05 definitions and released validation-share approximation. [All families and both samples](../artifacts/session_07/final_causal_ml_comparison.csv)

## Synthetic E policy comparison at 20%

Each policy selected 800 customers. The gain uses known CATE and represents expected incremental conversions.

| Policy | Mean true CATE (pp) | Expected incremental conversions | Negative-effect share | Gain shortfall to oracle |
| --- | ---: | ---: | ---: | ---: |
| Random | 1.300 | 10.40 | 52.50% | 117.60 |
| Predictive | 10.295 | 82.36 | 20.38% | 45.64 |
| Constant effect | 1.300 | 10.40 | 52.50% | 117.60 |
| S-logistic | 7.705 | 61.64 | 29.63% | 66.36 |
| S-LightGBM | 16.000 | 128.00 | 0.00% | 0.00 |
| T-logistic | 16.000 | 128.00 | 0.00% | 0.00 |
| T-LightGBM | 15.615 | 124.92 | 1.38% | 3.08 |
| X-linear | 16.000 | 128.00 | 0.00% | 0.00 |
| X-LightGBM | 16.000 | 128.00 | 0.00% | 0.00 |
| R-linear | 16.000 | 128.00 | 0.00% | 0.00 |
| R-LightGBM | 15.650 | 125.20 | 1.25% | 2.80 |
| LinearDML | 16.000 | 128.00 | 0.00% | 0.00 |
| Causal Forest | 16.000 | 128.00 | 0.00% | 0.00 |
| Oracle | 16.000 | 128.00 | 0.00% | 0.00 |

## Verification and artifacts

The full benchmark took 155.74 seconds, including the 30-repetition confounding study. The uncertainty tables were refreshed from the verified saved models in 7.38 seconds after adding direct checks for unsupported standard errors. Summed fit time was 11.04 seconds for LinearDML and 71.64 seconds for the forest. Training sizes, timing, fit configuration, inference profile and model provenance are recorded. Models are gitignored under `artifacts/session_07/models/`.

| Exact command | Result |
| --- | --- |
| `.venv/Scripts/python -m pip install --no-build-isolation -e '.[dev,advanced]'` | Installed the optional dependencies without replacing existing numerical libraries |
| `.venv/Scripts/python -m causallift.session07` | Full suite of benchmarks completed |
| `.venv/Scripts/python -m causallift.session07 --fixture --output data/processed/session07_fixture` | Offline fixture completed |
| `.venv/Scripts/python -m pytest -q` | 120 passed, zero failed or skipped with the optional dependency installed |
| `.venv/Scripts/python -m ruff check .` | Passed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

Hosted CI is pending publication approval. The optional-library tests skip in environments where EconML is absent. The local fixture flagged rank-deficient linear inference on its deliberately tiny data. Those intervals are withheld, while effect predictions remain available.

- [Structural DML](../artifacts/session_07/dml_estimation_results.csv) · [Confounding repetitions](../artifacts/session_07/high_dimensional_confounding.csv) · [Nuisance diagnostics](../artifacts/session_07/dml_diagnostics.json)
- [Synthetic accuracy](../artifacts/session_07/synthetic_effect_accuracy.csv) · [Ranking](../artifacts/session_07/uplift_ranking_metrics.csv) · [Combined benchmark](../artifacts/session_07/final_causal_ml_comparison.csv)
- [Targeting](../artifacts/session_07/targeting_comparison.csv) · [Calibration](../artifacts/session_07/uplift_calibration.csv) · [Forest intervals](../artifacts/session_07/causal_forest_uncertainty.csv) · [LinearDML intervals](../artifacts/session_07/linear_dml_uncertainty.csv)
- [Comparison plot](../artifacts/session_07/causal_ml_comparison.png) · [Model manifest](../artifacts/session_07/causal_model_manifest.json) · [Run summary](../artifacts/session_07/session_07_summary.json)

Session 08 can reuse fold-local nuisance fitting and effect benchmarks while establishing observational identification and overlap checks. The present confounding study illustrates why the observational diagnostics matter.
