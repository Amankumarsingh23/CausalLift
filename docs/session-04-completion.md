# CausalLift Session 04 completion report

## Predictive benchmark

The run trained 24 models across Criteo 200k, Criteo 1M and synthetic A, B, D and E. It fitted prevalence, scaled logistic regression and LightGBM for each configured outcome. Criteo used 120,000 training and 40,001 validation rows for 200k, and 600,000 training and 200,000 validation rows for 1M. Validation contained 117 and 583 conversions respectively. The two benchmark samples overlap.

| Dataset | Outcome | Model | ROC-AUC | Average precision | Log loss | Brier score | Fit seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 200k | Conversion | Prevalence | 0.5000 | 0.00292 | 0.019986 | 0.002916 | 0.037 |
| 200k | Conversion | Logistic | 0.9340 | 0.20786 | 0.012649 | 0.002576 | 0.262 |
| 200k | Conversion | LightGBM | 0.9495 | 0.19868 | 0.012534 | 0.002593 | 0.848 |
| 200k | Visit | Prevalence | 0.5000 | 0.04602 | 0.186651 | 0.043907 | 0.020 |
| 200k | Visit | Logistic | 0.9368 | 0.49504 | 0.108012 | 0.030406 | 0.511 |
| 200k | Visit | LightGBM | 0.9488 | 0.52650 | 0.100232 | 0.029030 | 1.414 |
| 1M | Conversion | Prevalence | 0.5000 | 0.00292 | 0.019928 | 0.002907 | 0.195 |
| 1M | Conversion | Logistic | 0.9494 | 0.21514 | 0.012313 | 0.002554 | 3.177 |
| 1M | Conversion | LightGBM | 0.9571 | 0.24608 | 0.011722 | 0.002473 | 4.543 |
| 1M | Visit | Prevalence | 0.5000 | 0.04682 | 0.189028 | 0.044623 | 0.180 |
| 1M | Visit | Logistic | 0.9318 | 0.47693 | 0.111453 | 0.031417 | 2.436 |
| 1M | Visit | LightGBM | 0.9460 | 0.50804 | 0.103304 | 0.029879 | 7.445 |

On 200k conversion, logistic achieved higher AP even though LightGBM led ROC-AUC. At 1M, LightGBM led all four reported metrics on both outcomes. On 1M conversion, observed prevalence was 0.2915%, compared with mean predictions of 0.2893% for logistic and 0.2923% for LightGBM. In the highest-score bins, logistic predicted 2.3557% versus 2.5750% observed (515 events among 20,000), and LightGBM predicted 2.5620% versus 2.6107% observed (522 among 19,995). Mean agreement alone does not establish reliability across the score range. No calibration model was fitted.

## Targeting at matched capacity

These are synthetic E validation results. Effects are based on known conditional treatment effects. Random and constant-effect selection coincide exactly because both use the same seeded selection order when scores are tied.

| Capacity | Selected | Policy | Mean CATE (pp) | Expected incremental conversions | Negative-effect fraction |
| --- | ---: | --- | ---: | ---: | ---: |
| 10% | 400 | Random / constant effect | +0.950 | +3.80 | 53.75% |
| 10% | 400 | Predictive | +13.060 | +52.24 | 10.50% |
| 10% | 400 | Oracle | +16.000 | +64.00 | 0.00% |
| 20% | 800 | Random / constant effect | +1.300 | +10.40 | 52.50% |
| 20% | 800 | Predictive | +10.295 | +82.36 | 20.38% |
| 20% | 800 | Oracle | +16.000 | +128.00 | 0.00% |
| 50% | 2,000 | Random / constant effect | +1.958 | +39.16 | 50.15% |
| 50% | 2,000 | Predictive | +8.188 | +163.76 | 27.90% |
| 50% | 2,000 | Oracle | +15.608 | +312.16 | 1.40% |

At 20% in E, the predictive group had a higher mean predicted conversion probability than the oracle group (52.21% versus 43.51%). The oracle still delivered 45.64 more expected incremental conversions. Across the existing scenarios, A tied because the effect is constant. B and D predictive rankings improved on random while trailing the oracle. No additional scenario was needed. Criteo results report only scores and observed composition for selected rows. Simulated sums are expectations, and the oracle requires unavailable counterfactual information on real data.

## Verification and model artifacts

Python 3.13.15 on Windows ran the benchmark. LightGBM 4.7.0 was added to the existing environment; the project compatibility range is `>=4.6,<5`. The initial benchmark emitted a deprecation warning for the validation input API. The final run selects the supported interface and completed without that warning. The manifest records exact versions. Total measured fit time was 3.09 seconds for 200k, 17.98 seconds for 1M and 0.65 seconds for the four synthetic cohorts together. These do not include report generation. The 24 Joblib files total about 481 KB under `artifacts/session_04/models/`.

The loaders returned only train and validation. Scaling used train only, and LightGBM early stopping used validation log loss. Tests verified predictor exclusions, fixed capacities and synthetic oracle evaluation after fitting. I reloaded all 24 saved models and independently reproduced their reported validation metrics and training fingerprints. The reserved test data was not used for fitting, stopping, calibration or evaluation.

| Exact command | Result |
| --- | --- |
| `.venv/Scripts/python -m pip install --no-build-isolation -e '.[dev]'` | Installed |
| `.venv/Scripts/python -m causallift.session04` | Both Criteo sizes and four synthetic scenarios completed |
| `.venv/Scripts/python -m causallift.session04 --fixture --output data/processed/session04_fixture` | Offline fixture reports generated |
| `.venv/Scripts/python -m pytest -q` | 92 passed, zero failed |
| `.venv/Scripts/python -m ruff check .` | Passed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

Hosted CI remains pending publication approval. Local model files and row-level data are gitignored.

## Outputs and Session 05 handoff

- [Criteo results](../artifacts/session_04/predictive_model_results.csv) · [Synthetic results](../artifacts/session_04/synthetic_predictive_results.csv) · [Calibration bins](../artifacts/session_04/calibration_summary.csv) · [Targeting metrics](../artifacts/session_04/targeting_baseline_results.csv)
- [Precision–recall](../artifacts/session_04/precision_recall_curves.png) · [ROC](../artifacts/session_04/roc_curves.png) · [Reliability](../artifacts/session_04/calibration_curves.png) · [Targeting chart](../artifacts/session_04/targeting_comparison.png)
- [Model manifest](../artifacts/session_04/predictive_model_manifest.json) · [Run summary](../artifacts/session_04/session_04_summary.json) · [Methods and reproduction](PREDICTIVE_BASELINES.md)

Session 05 can reuse the exact-capacity rankings, fixed assignments and synthetic truth joins to compare uplift estimators with these pooled predictive scores. Report validation comparisons with awareness of early stopping, the small number of Criteo positives, and overlap between benchmark samples.
