# CausalLift Session 06 completion report

## Implementation and data

The benchmark fitted and reloaded 32 X/R models. It used Criteo 200k, Criteo 1M and randomized synthetic A, B, D and E. The original benchmarks reloaded the Session 05 S/T artifacts and Session 04 predictive LightGBM scores without refitting them. The artifact audit matched all 60 prior ranking rows and checked equal targeting capacities and rank-invariant gain endpoints. Training and nuisance fitting used the fixed training rows. Validation supplied the development metrics, and test remained reserved. Synthetic truth was used only after fitting and scoring.

## Synthetic effect accuracy

Root PEHE and signed ATE error are in percentage points. Spearman is undefined for constant truth or constant predictions.

| Scenario | Learner | Root PEHE | Signed ATE error | Spearman |
| --- | --- | ---: | ---: | ---: |
| A | Constant effect | 0.379 | 0.379 | — |
| A | S-logistic | 1.013 | 0.234 | — |
| A | S-LightGBM | 4.326 | -0.467 | — |
| A | T-logistic | 1.645 | 0.261 | — |
| A | T-LightGBM | 9.250 | 0.568 | — |
| A | X-linear | 1.502 | 0.250 | — |
| A | X-LightGBM | 4.941 | 0.361 | — |
| A | R-linear | 1.499 | 0.282 | — |
| A | R-LightGBM | 7.027 | 0.551 | — |
| B | Constant effect | 8.661 | 0.536 | — |
| B | S-logistic | 8.603 | 0.550 | 0.724 |
| B | S-LightGBM | 5.118 | 0.136 | 0.853 |
| B | T-logistic | 4.773 | 0.516 | 0.882 |
| B | T-LightGBM | 8.975 | 0.669 | 0.665 |
| B | X-linear | 4.965 | 0.502 | 0.881 |
| B | X-LightGBM | 5.404 | 0.734 | 0.828 |
| B | R-linear | 4.963 | 0.498 | 0.881 |
| B | R-LightGBM | 7.538 | 0.634 | 0.723 |
| D | Constant effect | 8.726 | -0.262 | — |
| D | S-logistic | 8.842 | -0.716 | -0.657 |
| D | S-LightGBM | 5.988 | -0.641 | 0.797 |
| D | T-logistic | 6.525 | -1.228 | 0.757 |
| D | T-LightGBM | 13.038 | -0.745 | 0.542 |
| D | X-linear | 6.583 | -1.239 | 0.757 |
| D | X-LightGBM | 10.369 | -0.714 | 0.623 |
| D | R-linear | 6.602 | -1.250 | 0.756 |
| D | R-LightGBM | 11.087 | -0.809 | 0.554 |
| E | Constant effect | 14.028 | 0.914 | — |
| E | S-logistic | 13.898 | 0.830 | 0.380 |
| E | S-LightGBM | 5.479 | 0.650 | 0.807 |
| E | T-logistic | 8.886 | 1.013 | 0.797 |
| E | T-LightGBM | 9.276 | 0.659 | 0.767 |
| E | X-linear | 8.931 | 1.029 | 0.790 |
| E | X-LightGBM | 5.620 | 0.709 | 0.811 |
| E | R-linear | 8.925 | 1.036 | 0.791 |
| E | R-LightGBM | 7.886 | 0.705 | 0.784 |

A remained best served by the constant baseline. T-logistic had the lowest PEHE in B, and S-LightGBM led D and E. The X/R models used raw regression effects with no clipping. The validation range across all new models was −0.6794 to +0.8226, with zero values outside [−1, 1].

## Treatment imbalance

B and D share the conditional effect function. B had 5,951 treated and 6,050 control training units. D had 1,836 treated and 10,164 control training units. On the fixed splits, tree X/R improved on tree T in both scenarios. Under imbalance, the linear X/R models were close to T-logistic and S-LightGBM had the lowest error. All three tree estimates degraded from B to D.

| Tree learner | Fixed B PEHE | Fixed D PEHE | Repeated B mean (SD) | Repeated D mean (SD) |
| --- | ---: | ---: | ---: | ---: |
| T-LightGBM | 8.98 | 13.04 | 9.30 (0.20) | 12.93 (0.56) |
| X-LightGBM | 5.40 | 10.37 | 5.83 (0.38) | 9.96 (0.48) |
| R-LightGBM | 7.54 | 11.09 | 7.84 (0.52) | 10.80 (0.74) |

All errors in the table are in percentage points. Repeated seeds were 101, 202 and 303. The study resampled B and D with their original definitions and the matching seed per pair. T used its previous 100-tree limit and X/R used 65 trees for each component. The comparison is across configurations and simulation draws. [Study data](../artifacts/session_06/imbalance_study.csv) · [Imbalance plot](../artifacts/session_06/imbalance_comparison.png)

## Criteo 1M validation

Qini and AUUC keep the Session 05 formulas and scale. Uplift columns are percentage points.

| Outcome | Policy | Qini | AUUC | Uplift@10% | Uplift@20% | Uplift@50% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| conversion | Random | -0.0000296 | 0.0005450 | 0.116 | 0.091 | 0.091 |
| conversion | Predictive | 0.0003920 | 0.0011039 | 1.031 | 0.571 | 0.222 |
| conversion | Constant effect | -0.0000296 | 0.0005450 | 0.116 | 0.091 | 0.091 |
| conversion | S-logistic | 0.0003810 | 0.0010979 | 1.002 | 0.573 | 0.231 |
| conversion | S-LightGBM | 0.0004168 | 0.0011266 | 1.027 | 0.576 | 0.236 |
| conversion | T-logistic | 0.0003390 | 0.0010514 | 1.055 | 0.559 | 0.205 |
| conversion | T-LightGBM | 0.0000849 | 0.0007270 | 0.578 | 0.380 | 0.154 |
| conversion | X-linear | 0.0004171 | 0.0011343 | 1.106 | 0.541 | 0.236 |
| conversion | X-LightGBM | 0.0001797 | 0.0008513 | 0.794 | 0.411 | 0.185 |
| conversion | R-linear | 0.0004159 | 0.0011326 | 1.112 | 0.541 | 0.235 |
| conversion | R-LightGBM | 0.0002656 | 0.0009480 | 0.759 | 0.477 | 0.197 |
| visit | Random | 0.0003719 | 0.0065678 | 1.778 | 1.296 | 1.288 |
| visit | Predictive | 0.0036703 | 0.0112908 | 9.351 | 5.669 | 2.404 |
| visit | Constant effect | 0.0003719 | 0.0065678 | 1.778 | 1.296 | 1.288 |
| visit | S-logistic | 0.0035134 | 0.0111237 | 9.010 | 5.634 | 2.387 |
| visit | S-LightGBM | 0.0038515 | 0.0113680 | 10.159 | 5.610 | 2.391 |
| visit | T-logistic | 0.0037065 | 0.0114607 | 10.092 | 5.669 | 2.416 |
| visit | T-LightGBM | 0.0031806 | 0.0105862 | 9.657 | 5.482 | 2.199 |
| visit | X-linear | 0.0035918 | 0.0114220 | 9.955 | 5.522 | 2.413 |
| visit | X-LightGBM | 0.0032678 | 0.0108048 | 9.124 | 5.312 | 2.270 |
| visit | R-linear | 0.0035924 | 0.0114232 | 9.961 | 5.538 | 2.413 |
| visit | R-LightGBM | 0.0032237 | 0.0108036 | 10.137 | 5.256 | 2.280 |

Conversion had 583 validation events, including only 58 control events. S-LightGBM, X-linear and R-linear produced closely spaced Qini values. Different choices lead on ranking area and top-capacity uplift, so the differences are descriptive. Criteo fitting uses released training assignment shares. Evaluation retains the Session 05 validation-share approximation. True individual effects, original trial propensities and cluster identifiers are unavailable. The 200k and 1M cohorts overlap. [Uplift conventions](UPLIFT_EVALUATION.md)

## Synthetic E at 20% capacity

Every policy selected 800 validation rows. Incremental conversions are expectations from known conditional effects.

| Policy | Mean true CATE (pp) | Expected incremental conversions | Negative-effect share |
| --- | ---: | ---: | ---: |
| Random | 1.300 | 10.40 | 52.50% |
| Predictive | 10.295 | 82.36 | 20.38% |
| Constant effect | 1.300 | 10.40 | 52.50% |
| S-logistic | 7.705 | 61.64 | 29.63% |
| S-LightGBM | 16.000 | 128.00 | 0.00% |
| T-logistic | 16.000 | 128.00 | 0.00% |
| T-LightGBM | 15.615 | 124.92 | 1.38% |
| X-linear | 16.000 | 128.00 | 0.00% |
| X-LightGBM | 16.000 | 128.00 | 0.00% |
| R-linear | 16.000 | 128.00 | 0.00% |
| R-LightGBM | 15.650 | 125.20 | 1.25% |
| Oracle | 16.000 | 128.00 | 0.00% |

Both X configurations and R-linear matched oracle gain at this capacity, recovering 45.64 expected conversions over predictive targeting. R-LightGBM recovered 42.84. A tie in expected gains can include different units and yield different observed uplift.

## Cross-fitting and verification

All 16 R fits produced exactly one out-of-fold nuisance prediction per training row across three disjoint folds. Criteo 1M conversion R-LightGBM covered 600,000 training rows. Its out-of-fold log loss was 0.012072 and validation R-loss was 0.002464, versus 0.002451 with its frozen nuisance and a constant effect. Synthetic D R-linear covered 12,000 training rows, with out-of-fold log loss 0.617739 and validation R-loss 0.215658 versus 0.216322 for constant effect. Group-safe isolation and fold-local preprocessing were verified on fixtures.

The full benchmark took 70.17 seconds. Summed new fit time was 7.66 seconds for Criteo 200k, 37.81 seconds for Criteo 1M and 1.85 seconds for the four fixed synthetic cohorts. The run independently reloaded all new models and reproduced their validation effect predictions. The 32 Joblib binaries are gitignored under `artifacts/session_06/models/`.

| Exact command | Result |
| --- | --- |
| `.venv/Scripts/python -m causallift.session06` | Full benchmark and repeated-seed study completed |
| `.venv/Scripts/python -m causallift.session06 --fixture --output data/processed/session06_fixture` | Offline fixture completed |
| `.venv/Scripts/python -m pytest -q` | 112 passed, zero failed |
| `.venv/Scripts/python -m ruff check .` | Passed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

Python 3.13.15 and LightGBM 4.7.0 are unchanged. No dependencies were added. Hosted CI remains pending publication approval.

## Artifacts and Session 07 handoff

- [Estimator comparison](../artifacts/session_06/meta_learner_comparison.csv) · [Synthetic accuracy](../artifacts/session_06/synthetic_effect_accuracy.csv) · [Uplift ranking](../artifacts/session_06/uplift_ranking_metrics.csv)
- [Targeting](../artifacts/session_06/targeting_comparison.csv) · [Calibration](../artifacts/session_06/uplift_calibration.csv) · [Imbalance study](../artifacts/session_06/imbalance_study.csv) · [Imbalance plot](../artifacts/session_06/imbalance_comparison.png)
- [Cross-fit diagnostics](../artifacts/session_06/crossfit_diagnostics.json) · [Model manifest](../artifacts/session_06/causal_model_manifest.json) · [Run summary](../artifacts/session_06/session_06_summary.json)

Session 07 can reuse the deterministic fold assignments, nuisance-fitting callable, effect-only interface and fixed validation benchmark. The present real-data results are released-benchmark comparisons under approximate propensities. The small seed study characterizes sensitivity in these simulated designs. [Cross-fitting guide](CROSS_FITTING.md) · [Meta-learner guide](META_LEARNERS.md)
