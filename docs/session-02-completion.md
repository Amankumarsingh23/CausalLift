# CausalLift Session 02 completion report

## Implementation

Session 02 adds binary tests using pooled Z or Fisher exact inference, Newcombe and Wald difference intervals, Welch inference for continuous outcomes, optional stratified bootstrap intervals, and exploratory subgroup reports. The run reused Session 01 inputs and splits.

## Criteo v2.1

DuckDB computed exact control/treated counts, including 4,063/36,711 conversions and 80,105/576,824 visits in the complete 13,979,592-row archive. All comparisons use assigned treatment, pooled Z tests and 95% Newcombe intervals. Absolute changes and interval limits are in percentage points. [Criteo data documentation](DATASETS.md)

| Dataset | Outcome | Control / treated n | Control / treated rate | Absolute lift | Relative lift | 95% CI | Two-sided p |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| Full archive | Conversion | 2,096,937 / 11,882,655 | 0.1938% / 0.3089% | +0.1152 | +59.45% | [0.1084, 0.1219] | 7.31e−179 |
| Full archive | Visit | 2,096,937 / 11,882,655 | 3.8201% / 4.8543% | +1.0342 | +27.07% | [1.0055, 1.0628] | log₁₀(p) = −926.36 |
| 200k sample | Conversion | 30,000 / 170,000 | 0.1933% / 0.3088% | +0.1155 | +59.74% | [0.0536, 0.1672] | 0.000624 |
| 200k sample | Visit | 30,000 / 170,000 | 3.8100% / 4.8171% | +1.0071 | +26.43% | [0.7627, 1.2416] | 2.44e−14 |
| 1M sample | Conversion | 150,000 / 850,000 | 0.1940% / 0.3089% | +0.1149 | +59.25% | [0.0887, 0.1392] | 2.74e−14 |
| 1M sample | Visit | 150,000 / 850,000 | 3.7200% / 4.8609% | +1.1409 | +30.67% | [1.0338, 1.2461] | 9.68e−83 |

The full visit p-value is below floating-point range, so the report preserves its base-10 log. Privacy subsampling and pooled experiments limit interpretation to the released benchmark under its documented assumptions. Standard errors use independent rows because trial and user identifiers are unavailable. The cohort samples share observations and their outcome-stratified construction is outside the interval model.

## Synthetic benchmarks

Each scenario has 20,000 rows. Effects, intervals and signed errors are in percentage points. Error is the naive contrast minus true ATE. Scenario F’s truth uses the latent variable where appropriate.

| Scenario | Control / treated n | Control / treated rate | True ATE | True ATT | Naive contrast | 95% CI | Two-sided p | Error | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| A | 10,083 / 9,917 | 34.52% / 44.91% | +10.000 | +10.000 | +10.389 | [9.038, 11.735] | 5.78e−51 | +0.389 | Randomized |
| B | 10,083 / 9,917 | 34.52% / 35.00% | −0.066 | −0.128 | +0.477 | [−0.843, 1.797] | 0.479 | +0.544 | Randomized |
| C | 11,195 / 8,805 | 29.79% / 44.57% | −0.066 | +4.213 | +14.776 | [13.433, 16.113] | 3.28e−103 | +14.842 | Observational association |
| D | 16,940 / 3,060 | 34.39% / 33.95% | −0.066 | −0.136 | −0.438 | [−2.244, 1.403] | 0.639 | −0.371 | Randomized |
| E | 10,083 / 9,917 | 34.52% / 37.25% | +1.888 | +1.900 | +2.726 | [1.396, 4.054] | 5.86e−5 | +0.838 | Randomized |
| F | 10,003 / 9,997 | 33.80% / 53.61% | −0.057 | +5.907 | +19.806 | [18.454, 21.148] | 2.16e−175 | +19.863 | Observational association |

## Continuous outcomes and subgroups

The seeded synthetic revenue demonstration has 400 control and 300 treated observations. Means were 14.98 and 18.45. The estimated difference was +3.47 units, or +23.14% relative to control. Welch’s 95% CI was [2.05, 4.88], with t = 4.813, df = 503.88 and p = 1.96e−6. The bootstrap CI was [2.15, 4.85]. The known population difference is +2.00, which this realization’s intervals miss.

Validation-only subgroup analysis produced two Criteo `f0` bins and two synthetic E price-sensitivity bins. The high-`f0` Criteo bin had 18 total conversions and used Fisher’s exact test. The two synthetic E contrasts were −9.27 pp and +15.10 pp. These are exploratory, unadjusted results without a formal test of heterogeneity.

## Statistical verification and CI

Python 3.13.15 on Windows ran the complete suite. Statsmodels 0.15.0 was added only to development dependencies. All six real-data Z statistics, p-values and Newcombe intervals agreed with Statsmodels. Full-archive event totals also agreed with the Session 01 source profile. The suite cross-checks Welch’s statistic, degrees of freedom, interval and p-value against SciPy, and covers sparse and constant endpoints, bootstrap, aggregate parity and offline report generation.

| Local command | Result |
| --- | --- |
| `.venv/Scripts/python -m pip install --no-build-isolation -e '.[dev]'` | Installed successfully |
| `.venv/Scripts/python -m causallift.session02` | Full reports generated |
| `.venv/Scripts/python -m causallift.session02 --mode synthetic --output data/processed/session02_synthetic` | Synthetic-only reports generated |
| `.venv/Scripts/python -m causallift.session02 --mode fixture --output data/processed/session02_fixture` | Offline fixture reports generated |
| `.venv/Scripts/python -m ruff check .` | All checks passed |
| `.venv/Scripts/python -m pytest -q` | 58 passed, zero failed |
| `.venv/Scripts/python -m pip check` | No broken requirements |

The first pytest run had one failure from an unsupported keyword in a test assertion. The corrected test passes. Hosted CI is configured. The GitHub remote is empty and public, so running the workflow requires publishing the initial commit.

## Artifacts

- [Criteo results](../artifacts/session_02/criteo_experiment_results.csv)
- [Synthetic results](../artifacts/session_02/synthetic_experiment_results.csv)
- [Subgroup results](../artifacts/session_02/subgroup_analysis.csv)
- [Continuous demonstration](../artifacts/session_02/continuous_experiment_results.json)
- [Experiment summary](../artifacts/session_02/experiment_summary.json)
- [Treatment-effect intervals](../artifacts/session_02/treatment_effect_intervals.png)

## Session 03 handoff

Use the arm summaries and result records in `causallift.experimentation` as inputs for future experiment planning, health checks and multiplicity work. The current output demonstrates the gap between relative and absolute changes at a baseline conversion rate of 0.1938%. [Methods and reproduction guide](EXPERIMENTATION.md)
