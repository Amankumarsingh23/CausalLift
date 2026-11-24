# Classical experiment analysis

## Binary outcomes

The engine reports the treated rate minus the control rate as absolute lift. Multiply by 100 for percentage points. Relative lift divides absolute lift by the control rate and is undefined when control has zero events. A result includes both arm summaries, method names, test statistics, uncertainty, warnings and assumptions.

The two-sided Z-test uses the pooled event probability to compute the standard error under the null. It applies when all four expected counts under the null are at least five. Otherwise, the engine uses the two-sided Fisher exact test. Its statistic is the sample odds ratio when finite. The reported effect remains the rate difference. An all-zero or all-one table has Fisher p = 1 and a nonzero-width primary interval. A z-test p-value below floating-point range rounds to zero in the report. Use the accompanying `log10_p_value` for magnitude. [Statsmodels two-proportion test reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)

The primary binary confidence interval is Newcombe’s independent-arm difference interval. It combines the one-sided distances from Wilson limits in quadrature. The educational Wald interval uses the unpooled effect-estimation standard error and can extend outside the feasible range or collapse at the boundary. Fisher exact testing and Newcombe intervals can give different threshold decisions. [Statsmodels difference interval reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html)

Optional seeded percentile bootstrap resampling independently preserves each arm size. For binary outcomes the binomial draw with the observed arm rate is exactly the event-count distribution from resampling the arm’s rows. Runtime scales with the number of draws rather than the archive size. Very sparse outcomes can make the percentile bootstrap uninformative.

## Continuous outcomes

Welch’s two-sided test uses the arm sample variances, a standard error of `sqrt(s_t²/n_t + s_c²/n_c)` and Welch–Satterthwaite degrees of freedom for the p-value and interval. Each arm requires at least two finite outcomes. One constant arm is supported. When both arm variances are zero, the mean difference is available and Welch inference is undefined. Relative lift requires a positive control mean. The optional percentile bootstrap independently resamples the original rows in each arm. [SciPy Welch test reference](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

## Interpretation and analysis scope

Randomized assignment supports causal interpretation under design assumptions such as consistency, no interference and independent units for these standard errors. Synthetic scenarios C and F have observational assignment. Their p-values and intervals quantify association, including confounding. Scenario F’s true ATT averages full-information conditional effects among treated units.

Criteo uses assignment and includes all released units, regardless of exposure, for the intention-to-treat-style analysis. The full archive is checksum verified and read once by DuckDB for exact conversion and visit totals. The two subsamples are read from Parquet. Nonuniform privacy subsampling, pooled trials and missing trial/customer IDs limit inference to the released benchmark under the documented assumptions. The intervals use an independent-row approximation. The working datasets share source observations, and their outcome-stratified construction is not modeled in these intervals. [Criteo release](https://ailab.criteo.com/criteo-uplift-prediction-dataset/)

Overall results describe the full analysis cohorts. Subgroup analyses use only the validation partition, using the 200k training median for anonymized Criteo `f0` and the prespecified zero boundary for synthetic E’s price sensitivity. The subgroup helper accepts allowlisted categorical columns or predefined numeric bins. Missing-arm groups retain counts and rates for available arms and emit an insufficient-data warning. Subgroups are exploratory and p-values are unadjusted. Differences in significance between subgroups cannot establish heterogeneity.

## Reproduce the reports

Run from the repository root with the environment from [README.md](../README.md).

```powershell
.venv/Scripts/python -m causallift.session02
.venv/Scripts/python -m causallift.session02 --mode synthetic --output data/processed/session02_synthetic
.venv/Scripts/python -m causallift.session02 --mode fixture --output data/processed/session02_fixture
```

The default run reuses the complete Session 01 archive, the standard working Parquet, the optional one-million-row Parquet if present and six synthetic observed/truth pairs. The seed is 42, the confidence level is 0.95, and the bootstrap uses 2,000 draws. Override these with `--seed`, `--confidence-level` and `--bootstrap-resamples`. Use zero draws to disable bootstrap. `--root` selects the input tree and `--output` sets the report directory.

Call `analyze_binary(treatment, outcome)`, `binary_from_counts(treated_events, treated_n, control_events, control_n)` or `analyze_continuous(treatment, outcome)` in `causallift.experimentation` to reuse the engine. Use `binary_arm_counts(path, spec)` from `causallift.data.analytics` for Parquet sufficient statistics. [Measured results and verification](session-02-completion.md)
