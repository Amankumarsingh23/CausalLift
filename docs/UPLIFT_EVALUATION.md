# Uplift evaluation conventions

## Propensity-aware gain

For assigned treatment T, binary outcome Y and design probability e, the pseudo-outcome is `z = T × Y / e − (1 − T) × Y / (1 − e)`. Rank rows by descending score. Ties follow the same seeded random permutation for every policy on the cohort. Outcomes do not break ties. At prefix size k in a cohort of size N:

- Normalized cumulative gain `G(k) = sum_top_k(z) / N`.
- Uplift@K `U(k) = sum_top_k(z) / k` for supported, nonempty selections.
- Estimated incremental outcomes `N × G(k)`.
- Random-reference gain `(k/N) × G(N)`.
- AUUC is the trapezoidal integral of `G(k)` over `k/N`. Gain area above random integrates `G(k)` minus its random reference.

All rankings share the same full-population gain endpoint. Areas use every supported prefix before export to the compact curve CSV. Selections use floor-rounded 10%, 20% and 50% capacities. A prefix with one missing arm retains its raw cumulative curve coordinate but has no reported uplift estimate. Zero events with both arms yields point estimate zero and no normal interval.

## Within-prefix arm-count Qini

For each prefix with both arms, calculate `Q(k) = (treated_events(k) − control_events(k) × treated_n(k) / control_n(k)) / N`. The Qini coefficient integrates `Q(k) − (k/N) × Q(N)`. The origin is zero. Initial prefixes with either arm missing are undefined, so the trapezoid connects the origin to the first supported prefix. This prefix-adjusted convention is distinct from global-arm-ratio formulas and gain area above random. Both Qini and AUUC are in normalized outcome units on an x-axis from 0 to 1.

## Truth, calibration and uncertainty

Root PEHE is the square root of mean squared error against true CATE. CATE MAE averages absolute errors. Signed ATE error is mean predicted minus mean true effect; absolute ATE error takes its magnitude. Spearman is unavailable when either vector is constant to within 1e−12. Evaluation uses conditional effects rather than realized binary potential-outcome differences. True gain uses `sum_top_k(tau_true)/N`. The oracle is an evaluation-only ranking.

Calibration uses up to ten quantile bins, or four for Criteo conversion, keeping ties together. Each bin reports its predicted mean, arm and event counts, estimated uplift and synthetic true mean when available. Fewer than five units in either arm or fewer than five total events suppresses the calibration estimate. Sparse arm event counts flag uncertainty. Approximate pointwise intervals use the standard error of the pseudo-outcome mean, conditional on the scores and the supplied propensity, under independent-row assumptions. They omit cluster, propensity-estimation and model-fit uncertainty and do not compare policies formally.

Synthetic data uses the known design probabilities. Criteo uses the released validation treatment share as a working constant because original or stratum-specific propensities are unavailable. These curves describe the released benchmark under that approximation. Privacy subsampling, pooled trials and absent trial or unit IDs limit interpretation and uncertainty checks. A prefix’s empirical arm ratio is used only in the separate Qini calculation. Comparison samples overlap, and rare conversion counts make rankings noisy. [Dataset provenance](DATASETS.md)

## Reproduce

```powershell
.venv/Scripts/python -m causallift.session05
.venv/Scripts/python -m causallift.session05 --fixture --output data/processed/session05_fixture
```

Use `--development-only` to skip the optional 1M cohort. Full mode reads existing Session 01 datasets and Session 04 saved models. Use `--seed`, `--root` and `--output` for alternate runs. [Measured results](session-05-completion.md)
