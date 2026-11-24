# Session 11 completion

The full benchmark ran in 2.03 seconds and produced 423 policy comparisons on fixed validation cohorts. The final test split was not used. Values below are simulated contribution units. The benchmarks use seeded eligibility and the economic assumptions recorded in [economic scenarios](../artifacts/session_11/economic_scenarios.json).

## Representative synthetic E policies

| Cost design | Policy | Budget | Spend | Selected | True expected conversions | Predicted profit | True expected profit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | Random | 1,600 | 1,600 | 800 | 14.32 | −694.98 | −955.60 |
| Uniform | Predictive | 1,600 | 1,600 | 800 | 78.44 | 762.42 | 1,929.80 |
| Uniform | Profit | 1,600 | 1,600 | 800 | 128.00 | 3,905.22 | 4,160.00 |
| Uniform | Oracle | 1,600 | 1,600 | 800 | 128.00 | 2,776.04 | 4,160.00 |
| Variable, 180 rows | Uplift | 72 | 72 | 18 | 2.88 | 64.57 | 61.44 |
| Variable, 180 rows | Greedy profit | 72 | 72 | 28 | 4.48 | 133.14 | 168.48 |
| Variable, 180 rows | Greedy ratio | 72 | 71 | 32 | 5.12 | 138.69 | 185.16 |
| Variable, 180 rows | MILP | 72 | 72 | 32 | 5.12 | 138.73 | 179.68 |
| Variable, 180 rows | Oracle | 72 | 72 | 33 | 5.28 | 133.59 | 193.44 |

Uniform uplift and profit rankings tied in this E comparison. With variable value and uniform cost at budget 1,600, uplift gave true expected profit 4,095.20, profit ranking gave 5,620.64, and the oracle gave 6,024.48. Under unfavorable assumptions, every learned profit optimizer selected no treatment. The Session 04 targeting populations had different eligibility rules, so their counts at the same nominal capacity need not match.

All 12 learned and 12 oracle variable-cost solves were feasible and reported optimal with zero gap. On E at budget 72, the ratio heuristic took about 0.0002 s, learned MILP about 0.004 s, and oracle MILP about 0.008 s. The learned MILP's advantage in estimated profit did not translate to higher true expected profit. Its oracle regret was 13.76 compared with 8.28 for the ratio heuristic. Small fixture tests verified uniform and integer results against exhaustive enumeration and found a ratio-greedy case with objective 11 versus optimum 12. [Solver results](../artifacts/session_11/optimization_solver_results.csv) · [Budget frontier](../artifacts/session_11/budget_frontier.png)

## Discount and alternative treatments

Holding uplift at 10 pp, contribution at 40 and fixed delivery at 1, a conversion discount of 8 moved the high-baseline customer's profit from +3.00 to −3.40. The low-baseline customer moved from +3.00 to +1.40. Higher treated conversion probability increased incentive spending on conversions that would have happened anyway under the model.

In the 40-customer demonstration, the hypothetical-score MILP selected nine A actions, six B actions, and left 25 untreated. Spending was 36 at capacity 15. The estimated objective was 67.89 and true expected profit 94.26. The oracle achieved 94.26 as well, on the same constraints. [Discount calculations](../artifacts/session_11/discount_cost_demonstration.csv) · [Multi-treatment results](../artifacts/session_11/multi_treatment_results.csv)

## Artifacts, limitations and handoff

Scenario F is marked research-only. Scenarios B, D and E use their randomized design under stated assumptions, with overlap diagnostics marked unassessed where Session 10 had no entry. No actual spending, business recommendations or Criteo financial values are inferred. No offline policy estimator was added. Session 12 can reconstruct the learned assignments from the local Parquet and use formal policy evaluation with the recorded assumptions. [Methods](TREATMENT_OPTIMIZATION.md) · [Policy metrics](../artifacts/session_11/policy_optimization_results.csv) · [Reliability and provenance](../artifacts/session_11/policy_reliability_summary.json)
