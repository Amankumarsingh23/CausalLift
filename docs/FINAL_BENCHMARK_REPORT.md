# CausalLift benchmark report

The final replay covers the six cohorts and three representative model configurations named in the Session 12 protocol. The earlier experiments and model comparisons below keep their original populations. Every interval in the locked policy table is a fixed-action conditional 95% interval.

## Locked reserved-test evaluation

At the registered 20% setting, the synthetic E cohort had 3,999 rows and 1,435 events, including 696 control events. Criteo 1M had 200,000 rows and 583 conversions, including 58 control conversions. The 200k and 1M source benchmarks overlap.

| Cohort | Profit-aware versus | Paired DR profit | 95% interval | DR incremental conversions | Profit-aware / reference spend | True expected profit difference |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Synthetic E | Predictive | 286.82 | [−2,883.03, 3,456.67] | +6.37 | 1,598 / 1,598 | 2,179.80 |
| Synthetic E | Treat none | 3,583.05 | [1,194.73, 5,971.36] | +115.13 | 1,598 / 0 | 4,142.20 |
| Criteo 1M | Predictive | 11,117.61 | [4,858.33, 17,376.90] | −32.05 | 2,567.60 / 16,000 | Unavailable |
| Criteo 1M | Treat none | 8,736.82 | [−45.80, 17,519.44] | +103.52 | 2,567.60 / 0 | Unavailable |
| Synthetic A | Predictive | 1,735.22 | [−1,290.95, 4,761.39] | +38.56 | 1,598 / 1,598 | 0.00 |
| Synthetic B | Predictive | 2,256.09 | [128.32, 4,383.86] | +50.14 | 1,598 / 1,598 | 428.88 |
| Synthetic D | Predictive | 421.96 | [−2,953.88, 3,797.81] | +1.82 | 1,260 / 1,600 | 95.27 |
| Criteo 200k | Predictive | 1,338.30 | [−1,218.64, 3,895.25] | −17.50 | 310 / 3,199.60 | Unavailable |

Synthetic E selected 799 under both policies. Criteo 1M selected 6,419 profit-aware versus 40,000 predictive. Their paired profit SEs were 1,617.27 and 3,193.51 respectively. The Synthetic E advantage against predictive was inconclusive by DR despite positive simulator truth. Criteo's positive contrast against predictive coincided with lower spending and fewer estimated incremental conversions. Its interval against treat-none includes zero. The secondary comparisons are descriptive. [All paired estimates, support and standard errors](../artifacts/session_14/final_paired_policy_comparisons.csv)

| Cohort vs. predictive | Validation DR profit per row | Test DR profit per row | Test minus validation |
| --- | ---: | ---: | ---: |
| Synthetic E | 0.39256 | 0.07172 | −0.32084 |
| Criteo 1M | 0.03546 | 0.05559 | +0.02013 |
| Synthetic A | 0.17848 | 0.43391 | +0.25544 |
| Synthetic B | 0.22983 | 0.56416 | +0.33433 |
| Synthetic D | 0.51864 | 0.10549 | −0.41315 |
| Criteo 200k | 0.07985 | 0.03346 | −0.04640 |

[Matched comparison table](../artifacts/session_14/validation_vs_test.csv) · [Conditional intervals](../artifacts/session_14/final_policy_comparison.png) · [Validation to test chart](../artifacts/session_14/validation_vs_test.png)

Criteo uses the released test assignment share because original trial-level propensities are unavailable. All monetary amounts for Criteo are hypothetical. Unknown trial clustering, nuisance fitting uncertainty and allocation reoptimization are outside the intervals. Released-sample estimates cannot establish deployment performance. Scenario F remains research-only.

## Earlier benchmarks

| Area and population | Measured result and interpretation |
| --- | --- |
| Full data foundation | Verified 13,979,592 Criteo rows, sampled 200k and 1M working benchmarks, and generated six 20k synthetic scenarios. The working samples overlap. [Data report](session-01-completion.md) |
| A/B on the released archive | Conversion assignment difference +0.1152 pp, 95% CI [0.1084, 0.1219]. Independent-row intervals omit unknown trial clustering. [Experimentation](session-02-completion.md) |
| Prospective design | A 10% relative conversion lift at 85% treated and 80% power required about 3.33M rows at the illustrative Criteo baseline. [Planning](session-03-completion.md) |
| Predictive validation | Pooled LightGBM Criteo 1M conversion ROC-AUC 0.9571 and AP 0.24608. The 200k logistic baseline had higher AP than LightGBM. [Predictive modeling](session-04-completion.md) |
| Meta-learners | Synthetic E S-LightGBM root PEHE 5.479 pp versus 14.028 pp for constant effect. On Criteo 1M conversion, validation R-loss for R-LightGBM was 0.002464 versus 0.002451 for constant effect. [S/T](session-05-completion.md) · [X/R](session-06-completion.md) |
| Causal Forest and DML | Forest and LinearDML compared under common cohorts. Partially linear DML estimated synthetic E training ATE at 2.635 pp versus truth 1.832 pp. High-dimensional DML intervals covered truth in 0/30. [Benchmark](session-07-completion.md) |
| Observational validation | Scenario C AIPW −1.75 pp versus truth −0.08 pp, with max weight 110.71 and ESS 1,246 of 4,000. Scenario F remained biased. [Identification](session-08-completion.md) · [Estimates](session-09-completion.md) |
| Reliability | Scenario A matching changed from −55.09 pp with one reused control to +10.38 pp with outcome-independent tie handling. Scenario F stable subset estimates remained biased. [Refutation and correction](session-10-completion.md) |
| Optimization | Synthetic E variable-cost greedy expected truth 185.16, learned MILP 179.68 and oracle 193.44. The learned objective was estimated profit. [Optimization](session-11-completion.md) |
| Offline validation | Synthetic E profit-aware fixed-action true expected profit 4,160 at a 1,600 budget. Criteo 1M paired DR advantage was 7,091.50 under hypothetical economics and unequal spending. [Offline evaluation](session-12-completion.md) |
| Platform | Seven browser-verified views. Fixture budget 60 selected 30 customers and generated modeled profit 114.22. [Platform verification](session-13-completion.md) |
| Locked final replay | Six cohorts and 18 used model binaries verified. The registry exposes only published aggregate results. [Integrity manifest](../artifacts/session_14/final_evaluation_manifest.json) |
