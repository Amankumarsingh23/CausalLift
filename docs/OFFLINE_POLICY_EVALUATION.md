# Offline policy evaluation

Evaluate fixed binary actions on the same ordered cohort. For action `a`, logged treatment `T`, binary outcome `Y`, and logging treatment probability `e`, define `w = I(T = a) / (e if T = 1 else 1 - e)`. The outcome models `mu0` and `mu1` are fitted on training rows.

| Estimator | Policy outcome rate |
| --- | --- |
| Direct method (DM) | Mean of `mu[a]` |
| Inverse propensity scoring (IPS) | Mean of `w * Y` |
| Self-normalized IPS (SNIPS) | Sum of `w * Y` divided by sum of `w` |
| Doubly robust (DR) | Mean of `mu[a] + w * (Y - mu[T])` |

For DM, IPS and DR, difference the scores row by row against treat-none or a second fixed policy. The mean difference estimates the incremental outcome rate and the sum estimates incremental conversions. DR treat-all minus treat-none reduces to the existing AIPW ATE on identical rows and nuisance predictions. SNIPS reports outcome rates only and is undefined when no logged actions match. Zero support and concentrated weights are flagged.

For additive scores, incremental profit for policy `a` relative to `b` is the sum of `contribution * (score[a] - score[b]) - cost * (a - b)`. The proposed cost is known and enters directly. ROI against treat-none divides profit by proposed spending and is undefined at zero spend. These formulas assume a cost incurred for each proposed treatment. Conversion-contingent discounts require a different cost calculation. Synthetic truth uses `tau * (a - b)` and measures expected change.

Approximate 95% Wald intervals use the variability of paired row scores, conditional on the fixed action vectors and nuisance predictions. They omit reoptimization on new cohorts, nuisance fitting and trial clustering. Missing matching support suppresses IPS and DR intervals. Scenario F retains its observational identification limitations and receives no policy intervals. A positive estimate cannot resolve unmeasured confounding or positivity failure.

Criteo uses the released cohort treatment share as a propensity approximation. Original trial propensities and business economics are unavailable, and rare conversions limit precision. The released-sample analysis uses hypothetical contribution and cost inputs.

```powershell
python -m causallift.session12
python -m causallift.session12 --fixture --output <directory>
```

The full run verifies the Session 11 allocation provenance and saves tables and figures under `artifacts/session_12/`. The [frozen protocol](../artifacts/session_12/final_evaluation_protocol.json) defines Session 14 cohorts, saved model fingerprints, assumptions, budgets, seeds, and metrics. Build and freeze test action vectors from ordered IDs and pre-treatment features before joining outcomes, logged treatment, or synthetic truth. Session 12 used training and validation data.
