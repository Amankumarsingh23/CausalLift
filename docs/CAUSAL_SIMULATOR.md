# Synthetic causal simulator

The shared NumPy-generator pipeline draws six independent standard normal pre-treatment covariates. Baseline outcome probability starts with a logistic index of engagement, loyalty, spend and recency. For A–E, the baseline is `0.20 + 0.45 × logistic(index)`. The treated probability is baseline plus the scenario effect. Effects are bounded to keep both probabilities valid without clipping. The simulator draws Bernoulli potential outcomes independently conditional on all covariates, samples assignment at its known propensity, then selects the corresponding potential outcome.

| Scenario | Assignment | Probability-scale effect |
| --- | --- | --- |
| A | 50% randomized | +0.10 everywhere |
| B | 50% randomized | `0.12 sin(engagement) + 0.07 tanh(price_sensitivity × activity)` |
| C | Observed engagement and loyalty affect assignment and baseline | Same nonlinear effect as B |
| D | 15% randomized | Same nonlinear effect as B |
| E | 50% randomized | +0.16 for nonnegative price sensitivity and −0.12 otherwise |
| F | Engagement and a latent standard normal variable affect assignment, with many propensities near zero or one | Nonlinear effect from B plus a centered latent term |

In F the same latent variable also shifts the baseline index, with baseline probability `0.27 + 0.38 × logistic(index + 1.5 × latent)`. The additional effect is `0.06 × (2 × normal_cdf(latent) − 1)`. Its conditional mean given the observable covariates is zero because the latent variable is independent of those covariates. F therefore stores both `tau_true` conditional on all covariates and `tau_observed_true` marginal over the latent variable. Observable-covariate exchangeability fails, so observed-data adjustment alone cannot identify that marginal CATE without further assumptions.

The observed files contain covariates, scenario-local unit ID, assignment, outcome and split. The evaluation files contain probabilities, true propensity, realized potential outcomes and true CATE (`mu1_true − mu0_true`). The realized contrast `y1_true − y0_true` can differ from CATE. True ATE averages CATE over all generated units. True ATT averages full-information CATE over assigned treated units, including in F. Naive difference compares observed outcome means by arm.

Run the default seed and 20,000 rows per scenario with the commands in [README.md](../README.md). The same seed gives scenarios paired covariate draws for controlled comparisons. IDs are scenario-local. Treating scenarios as independent replicates would require different seeds. Run settings and measured metrics are recorded in the generated artifacts.
