# Experiment design and health

## Power, sample size and minimum detectable effect

Type I error is the chance of rejecting a true null. Type II error is the chance of missing a specified alternative. Power is one minus Type II error at that alternative. A minimum detectable effect (MDE) is the effect that reaches a chosen power at fixed sample sizes and alpha.

Binary planning uses Cohen’s variance-stabilized effect, `h = 2 × (arcsin(sqrt(p_t)) − arcsin(sqrt(p_c)))`. Define effective size as `1 / (1/n_c + 1/n_t)` and the alternative shift as `abs(h) × sqrt(effective size)`. The two-sided power is the probability that a unit-variance normal with that shift falls beyond either null critical boundary. This is a planning approximation. Session 02 inference uses the pooled-proportion Z or Fisher test on actual outcomes. [Statsmodels effect-size reference](https://www.statsmodels.org/v0.14.6/generated/statsmodels.stats.proportion.proportion_effectsize.html) · [Statsmodels normal power reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.power.NormalIndPower.power.html)

`binary_sample_size` accepts an alternative rate, signed absolute effect or signed relative effect. Its ratio is treated divided by control. With a ratio of `r`, continuous arm sizes are effective size × `(1 + 1/r)` for control and × `(1 + r)` for treatment. The planner rounds upward and rechecks power on integer counts. `binary_mde` inverts the same formula in the requested direction and reports the alternative, signed absolute difference and signed relative change. Requests that cannot attain target power within probability bounds raise an error. Sparse cells trigger warnings in sample-size and MDE results. Use `binary_power_warnings` alongside power calculations at fixed sizes. Extreme rare-event settings can require a design-specific simulation.

`continuous_sample_size` and `continuous_power` replace Cohen’s h with the mean difference divided by a common population standard deviation. They use a large-sample normal approximation with that shared variance assumption. Criteo baselines come from exact Session 02 counts. The plan treats allocation, alpha and MDE as hypothetical design settings for future experiments. Post-hoc power adds no evidence to the already observed Criteo effects.

## Assignment and balance

SRM uses a Pearson goodness-of-fit statistic against the explicit intended share, with one degree of freedom and an exact two-sided binomial fallback when either expected count is below five. The default alert threshold is 0.01. Missing intended assignment produces a `not_assessable` result. The released Criteo share is descriptive. An SRM alert merits investigation of assignment, eligibility, joins or logging. The intended design should make the assignment-count null and its binomial approximation appropriate.

Balance reports use `(mean_t − mean_c) / sqrt((variance_t + variance_c) / 2)` with sample variances and equal weighting of the two variances. The default flag is `abs(SMD) >= 0.1`. Both arms constant makes SMD undefined. Unequal constants are flagged and equal constants are reported without a flag decision. The examples use three allowlisted features per dataset on training rows. Balance covers measured covariates only and cannot rule out hidden confounding.

## Multiplicity and guardrails

Bonferroni and Holm control family-wise error under arbitrary dependence when individual p-values are valid. Benjamini–Hochberg targets the expected false discovery proportion under independence or suitable positive dependence. Every tested subgroup for a dataset and feature belongs to one prespecified validation-only family. Missing tests occupy a family slot with p = 1 for adjustment, and their results are unavailable. Under the global null any rejection is false, so FDR equals the probability of any rejection. That equality is specific to the global null. [Statsmodels adjustment reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)

Decisions orient each effect so larger means better. The primary metric passes when the lower benefit limit exceeds the practical threshold and its two-sided p-value is below alpha. It fails when the upper benefit limit is at or below the threshold. Each mandatory guardrail passes when its lower benefit limit exceeds the negative harm margin and fails when its upper limit is strictly below that margin. Missing metrics or intervals that straddle a boundary leave the evidence inconclusive. All decision intervals use the prespecified confidence level. This uses the lower endpoint of a two-sided interval for safety, which is conservative relative to a one-sided test at the same alpha.

Supplied failed or unassessable health takes precedence and returns `investigate`. A failure returns `do_not_ship`, all passes return `ship`, and remaining cases return `inconclusive`. Omitting health leaves the metric rules to decide. Prespecify directions, thresholds, margins and the analysis population. The deterministic case studies use synthetic conversion and refund data. The policy does not supply simultaneous confidence intervals or a general sequential decision rule.

## CUPED and simulations

CUPED applies `Y_adjusted = Y − theta × (X − pilot_mean_X)` with `theta = Cov_pilot(X, Y) / Var_pilot(X)`. It uses a pre-experiment covariate and an independent pilot. In the evaluation sample, Session 02 Welch inference treats the pilot slope as fixed. The reported interval is conditional on the pilot and the repeat simulations hold that pilot fixed. A constant pilot covariate results in no adjustment. One realized reduction is `1 − adjusted_SE² / raw_SE²`, distinct from variance reduction measured across repetitions. Weak correlation can yield little improvement, and a poorly estimated or transported slope can increase variance. [Deng et al., CUPED](https://ai.stanford.edu/~ronnyk/2013-02CUPEDImprovingSensitivityOfControlledExperiments.pdf)

The default run uses seed 42 and 3,000 repetitions per check. Binary simulations use binomial event totals, including ten cumulative equally spaced inspections and a matching final horizon under the null. The alternative uses a control rate of 0.10 and treated rate of 0.12 at the planned arm sizes. Coverage uses Session 02 Newcombe intervals. Multiplicity uses 20 independent uniform null p-values per family. CUPED uses the exact Gaussian arm-mean-difference distribution across independent experiments conditional on the shared pilot. The report provides Monte Carlo standard errors and underlying parameters. The plot error bars show approximately 1.96 Monte Carlo standard errors. Peeking is an illustration of error inflation under the specified schedule.

## Reproduce

```powershell
.venv/Scripts/python -m causallift.session03
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pip check
```

Use `--root` to select existing input artifacts and `--output` for a separate result directory. Override `--seed` and `--repetitions` for new simulations. Inputs are Session 02’s Criteo and subgroup CSVs and Session 01’s Criteo and synthetic A/C Parquet files. No Criteo archive download or scan is needed. The report preserves the train/validation/test boundaries. [Measured results and output paths](session-03-completion.md)
