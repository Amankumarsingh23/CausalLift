# CausalLift Session 05 completion report

Session 05 fitted 32 S/T models across the four randomized synthetic scenarios and both Criteo benchmarks. It reused the saved Session 04 LightGBM baselines and the existing data splits. The run evaluated conditional-effect accuracy, normalized uplift gain, separate Qini, calibration and targeting under equal capacity.

## Synthetic effect accuracy

Root PEHE and signed ATE error are shown in percentage points. A blank correlation indicates constant truth or constant predictions.

| Scenario | Learner and base | Root PEHE (pp) | Signed ATE error (pp) | Spearman |
| --- | --- | ---: | ---: | ---: |
| A | Constant effect | 0.379 | 0.379 | — |
| A | S logistic | 1.013 | 0.234 | — |
| A | S LightGBM | 4.326 | -0.467 | — |
| A | T logistic | 1.645 | 0.261 | — |
| A | T LightGBM | 9.250 | 0.568 | — |
| B | Constant effect | 8.661 | 0.536 | — |
| B | S logistic | 8.603 | 0.550 | 0.724 |
| B | S LightGBM | 5.118 | 0.136 | 0.853 |
| B | T logistic | 4.773 | 0.516 | 0.882 |
| B | T LightGBM | 8.975 | 0.669 | 0.665 |
| D | Constant effect | 8.726 | -0.262 | — |
| D | S logistic | 8.842 | -0.716 | -0.657 |
| D | S LightGBM | 5.988 | -0.641 | 0.797 |
| D | T logistic | 6.525 | -1.228 | 0.757 |
| D | T LightGBM | 13.038 | -0.745 | 0.542 |
| E | Constant effect | 14.028 | 0.914 | — |
| E | S logistic | 13.898 | 0.830 | 0.380 |
| E | S LightGBM | 5.479 | 0.650 | 0.807 |
| E | T logistic | 8.886 | 1.013 | 0.797 |
| E | T LightGBM | 9.276 | 0.659 | 0.767 |

The constant baseline had the lowest root PEHE in homogeneous A. T-logistic led B, and S-LightGBM led D and E. D used 1,836 treated and 10,164 control training rows. Its T-LightGBM root PEHE rose to 13.04 pp from 8.98 pp on B, which uses the same underlying effect function with more balanced assignment. This is a benchmark comparison across two realized samples.

## Criteo ranking metrics

Qini is the baseline-adjusted area of the within-prefix arm-count curve, normalized by the evaluation population size. AUUC is the raw area of propensity-weighted gain per eligible unit. Uplift columns are percentage points and use the released validation assignment share as a constant propensity approximation.

| Cohort | Outcome | Learner and base | Qini | AUUC | Uplift@10% (pp) | @20% (pp) | @50% (pp) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Criteo 200k | conversion | S logistic | 0.0004694 | 0.0011413 | 1.736 | 0.775 | 0.194 |
| Criteo 200k | conversion | S LightGBM | 0.0004242 | 0.0011059 | 1.059 | 0.559 | 0.229 |
| Criteo 200k | conversion | T logistic | 0.0004417 | 0.0011049 | 1.167 | 0.613 | 0.224 |
| Criteo 200k | conversion | T LightGBM | 0.0002175 | 0.0008430 | 1.020 | 0.446 | 0.157 |
| Criteo 200k | visit | S logistic | 0.0025813 | 0.0090211 | 7.583 | 4.689 | 1.940 |
| Criteo 200k | visit | S LightGBM | 0.0024700 | 0.0085538 | 7.524 | 4.316 | 1.823 |
| Criteo 200k | visit | T logistic | 0.0037476 | 0.0100043 | 9.719 | 5.296 | 2.134 |
| Criteo 200k | visit | T LightGBM | 0.0015247 | 0.0073710 | 4.886 | 3.855 | 1.605 |
| Criteo 1M | conversion | S logistic | 0.0003810 | 0.0010979 | 1.002 | 0.573 | 0.231 |
| Criteo 1M | conversion | S LightGBM | 0.0004168 | 0.0011266 | 1.027 | 0.576 | 0.236 |
| Criteo 1M | conversion | T logistic | 0.0003390 | 0.0010514 | 1.055 | 0.559 | 0.205 |
| Criteo 1M | conversion | T LightGBM | 0.0000849 | 0.0007270 | 0.578 | 0.380 | 0.154 |
| Criteo 1M | visit | S logistic | 0.0035134 | 0.0111237 | 9.010 | 5.634 | 2.387 |
| Criteo 1M | visit | S LightGBM | 0.0038515 | 0.0113680 | 10.159 | 5.610 | 2.391 |
| Criteo 1M | visit | T logistic | 0.0037065 | 0.0114607 | 10.092 | 5.669 | 2.416 |
| Criteo 1M | visit | T LightGBM | 0.0031806 | 0.0105862 | 9.657 | 5.482 | 2.199 |

S-LightGBM led 1M conversion Qini among the causal learners. Its Qini was 0.000417 versus 0.000392 for the saved predictive ranking. These are close benchmark estimates without a formal difference test. There were only 117 validation conversions on 200k and 583 on 1M (58 control and 525 treated on 1M). Criteo 200k had 35 control conversions in training, versus 175 on 1M. The calibration rules withheld 8 conversion bins on 200k and 3 on 1M across the reported effect scores. Sparse groups that retained estimates include uncertainty warnings where applicable.

Criteo’s privacy subsampling and missing trial or customer IDs limit interpretation. The validation assignment fraction is estimated from the release. It is not the verified original randomization probability. The cohorts overlap and intervals assume independent rows with fixed scores and propensity.

## Synthetic E at 20% capacity

| Policy | Selected | Mean true CATE (pp) | Expected incremental conversions | Negative-effect share | Predictive-to-oracle gap closed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 800 | 1.300 | 10.40 | 52.50% | — |
| Predictive | 800 | 10.295 | 82.36 | 20.38% | — |
| Constant effect | 800 | 1.300 | 10.40 | 52.50% | — |
| S logistic | 800 | 7.705 | 61.64 | 29.63% | -45.4% |
| S LightGBM | 800 | 16.000 | 128.00 | 0.00% | 100.0% |
| T logistic | 800 | 16.000 | 128.00 | 0.00% | 100.0% |
| T LightGBM | 800 | 15.615 | 124.92 | 1.38% | 93.3% |
| Oracle | 800 | 16.000 | 128.00 | 0.00% | — |

S-LightGBM and T-logistic matched the oracle’s true gain at this capacity. Different selected units can have the same true CATE and different observed outcomes. That explains why these policies can tie on true gain while their observed uplift estimates differ. T-LightGBM closed 93.3% of the 45.64-conversion gap, and S-logistic widened it by 20.72 expected conversions.

## Verification, artifacts and handoff

The full run took 67.99 s. Summed causal fit time was 6.65 s for 200k, 29.50 s for 1M and 1.75 s across the four synthetic cohorts. All 32 Joblib models were reloaded and both potential-outcome vectors matched. The files live in `artifacts/session_05/models/` and are gitignored.

Training used allowlisted features and assigned treatment. S added an explicit assignment flag. T used assignment only for arm selection. Truth was loaded after fitting and validation predictions, for synthetic evaluation. Final test partitions stayed reserved. The full-artifact audit confirmed capacity parity, curve origins and endpoints, and agreement with Session 04 synthetic policy gains.

| Command | Local result |
| --- | --- |
| `.venv/Scripts/python -m causallift.session05` | Both Criteo benchmarks and four synthetic scenarios completed |
| `.venv/Scripts/python -m causallift.session05 --fixture --output data/processed/session05_fixture` | Offline fixture completed |
| `.venv/Scripts/python -m pytest -q` | 102 passed, zero failed |
| `.venv/Scripts/python -m ruff check .` | Passed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

The environment remained Python 3.13.15 and LightGBM 4.7.0 with no new dependencies. Hosted CI remains pending publication approval.

- [Estimator results](../artifacts/session_05/causal_estimator_results.csv) · [Synthetic accuracy](../artifacts/session_05/synthetic_effect_accuracy.csv) · [Ranking metrics](../artifacts/session_05/uplift_ranking_metrics.csv)
- [Targeting comparisons](../artifacts/session_05/targeting_comparison.csv) · [Calibration bins](../artifacts/session_05/uplift_calibration.csv) · [Curve coordinates](../artifacts/session_05/uplift_curve_coordinates.csv)
- [Gain plot](../artifacts/session_05/uplift_curves.png) · [Qini plot](../artifacts/session_05/qini_curves.png) · [Calibration plot](../artifacts/session_05/uplift_calibration.png)
- [Model manifest](../artifacts/session_05/causal_model_manifest.json) · [Run summary](../artifacts/session_05/session_05_summary.json)

Session 06 can reuse the effect-only estimator interface, fixed validation populations and observed and synthetic evaluation metrics to study imbalance and cross-fitting. [Learner guide](META_LEARNERS.md) · [Evaluation conventions](UPLIFT_EVALUATION.md)
