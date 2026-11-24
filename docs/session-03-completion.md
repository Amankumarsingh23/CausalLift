# CausalLift Session 03 completion report

## Implementation

Session 03 adds prospective binary and continuous planning, SRM, standardized covariate balance, multiplicity adjustments, guardrail decisions, independent-pilot CUPED and Monte Carlo checks. It reuses the Session 01 data and Session 02 event counts and interval engine. [Methods and reproduction](EXPERIMENT_DESIGN.md)

## Planning with Criteo control rates

The matrix contains 24 plans across two outcomes, two allocations, two power targets and three relative effects. The 16 fixed-size MDE rows cover 200k and 1M designs. Examples below use 5% alpha, 80% power and a +10% relative effect. Shares refer to treatment.

| Outcome | Baseline | Target rate | Treated share | Control n | Treated n | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conversion | 0.1938% | 0.2131% | 50% | 848,466 | 848,466 | 1,696,932 |
| Conversion | 0.1938% | 0.2131% | 85% | 499,098 | 2,828,217 | 3,327,315 |
| Visit | 3.8201% | 4.2021% | 50% | 41,394 | 41,394 | 82,788 |
| Visit | 3.8201% | 4.2021% | 85% | 24,350 | 137,979 | 162,329 |

All listed plans attain at least their target after rounding. At 85% treated and 80% power, conversion MDEs are +0.0848 pp at 200k (+43.77% relative) and +0.0360 pp at 1M (+18.60%). Visit MDEs are +0.3434 pp at 200k (+8.99%) and +0.1518 pp at 1M (+3.97%). Continuous planning at a standardized 0.2 effect and 50:50 allocation needs 393 per arm at 80% power. These are approximate prospective designs, using the released benchmark rates as illustrative inputs.

## Health, multiplicity and guardrails

| SRM case | Observed treated | Intended | p-value | Result |
| --- | ---: | ---: | ---: | --- |
| Healthy synthetic | 490 / 1,000 | 50% | 0.527 | No mismatch detected |
| Mismatched synthetic | 350 / 1,000 | 50% | 2.38e−21 | Mismatch |
| Unequal synthetic | 150 / 1,000 | 15% | 1.000 | No mismatch detected |
| Full Criteo release | 11,882,655 / 13,979,592 | Unavailable | — | Not assessable |

The absolute SMD threshold of 0.1 flagged none of the three selected features in synthetic A or Criteo training. Synthetic C flagged engagement (1.106) and loyalty (0.648). These diagnostics characterize measured covariates.

| Validation subgroup family | Raw | Bonferroni | Holm | Benjamini–Hochberg |
| --- | ---: | ---: | ---: | ---: |
| Criteo `f0` | 0 | 0 | 0 | 0 |
| Synthetic E price sensitivity | 2 | 2 | 2 | 2 |

The four synthetic guardrail cases returned `ship` when conversion improved and refund safety was established, `do_not_ship` when refund harm exceeded the margin, `inconclusive` when safety remained uncertain, and `investigate` when health failed.

## CUPED

| Correlation case | Pilot correlation | Raw effect (SE) | Adjusted effect (SE) | Adjusted 95% CI | Realized variance reduction | Simulated variance reduction |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| Strong | 0.896 | 0.501 (0.137) | 0.414 (0.064) | [0.289, 0.540] | 78.41% | 80.36% |
| Weak | 0.099 | 0.399 (0.065) | 0.399 (0.064) | [0.275, 0.524] | 2.88% | 1.93% |

The true effect is +0.4 in both examples. The independently fitted pilot slope was held fixed during evaluation. Raw and adjusted intervals and pilot details are in the CUPED report.

## Monte Carlo verification

Seed 42, 3,000 repetitions per check. SEs are Monte Carlo standard errors in percentage points. Family-level results use 20 independent true nulls. Under this design FDR and FWER coincide because all rejections are false.

| Check | Reference | Empirical | MC SE |
| --- | ---: | ---: | ---: |
| Fixed-horizon Type I error | 5.00% | 4.97% | 0.40 |
| Naive peeking, ten inspections | — | 19.10% | 0.72 |
| Power, 10% versus 12% | 80.00% | 80.23% | 0.73 |
| Newcombe interval coverage | 95.00% | 94.47% | 0.42 |
| Uncorrected FWER | 64.15% | 62.57% | 0.88 |
| Bonferroni FWER | 4.88% | 4.57% | 0.38 |
| Holm FWER | 4.88% | 4.57% | 0.38 |
| Benjamini–Hochberg FDR | 5.00% | 4.63% | 0.38 |

The simulations use a single binary baseline and effect, one peeking schedule and independently generated null families. Rare events, correlation between hypotheses and other schedules can change empirical behavior.

## Verification

Local environment was Python 3.13.15 on Windows. The session added no dependencies. Statsmodels validates planning and multiplicity and SciPy independently validates SRM. Tests also cover guardrail direction, CUPED, deterministic simulation and offline report generation.

| Exact command | Result |
| --- | --- |
| `.venv/Scripts/python -m causallift.session03` | 11 artifacts generated |
| `.venv/Scripts/python -m pytest -q` | 83 passed, zero failed |
| `.venv/Scripts/python -m ruff check .` | All checks passed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

The initial suite caught a Benjamini–Hochberg boundary rounding discrepancy. The corrected version compares sorted p-values to their rank thresholds for decisions and passes the reference fixture. Hosted CI is pending initial publication approval for the empty public remote.

## Artifacts and handoff

- Planning and health · [Sample sizes](../artifacts/session_03/experiment_planning.csv) · [MDE](../artifacts/session_03/mde_analysis.csv) · [SRM](../artifacts/session_03/srm_results.json) · [Balance data](../artifacts/session_03/covariate_balance.csv) · [Balance plot](../artifacts/session_03/covariate_balance.png)
- Decisions · [Adjusted subgroup tests](../artifacts/session_03/multiple_testing_results.csv) · [Guardrails](../artifacts/session_03/guardrail_decisions.json) · [CUPED](../artifacts/session_03/cuped_results.json)
- Simulations · [Results](../artifacts/session_03/monte_carlo_results.csv) · [Plot](../artifacts/session_03/statistical_simulation_summary.png) · [Design summary](../artifacts/session_03/experiment_design_summary.json)

Session 04 can use the existing Session 01 training, validation and test partitions and feature allowlists for predictive baselines. Session 02 and 03 provide the aggregate assignment benchmark and explain why predictive conversion performance and causal effects answer different questions.
