# Propensity scores and overlap

Propensity predicts **assignment** from the pre-treatment allowlist. The comparison includes a training-share constant, scaled logistic regression, and a fixed 55-tree LightGBM. Three deterministic folds fit only within training. Stratification uses assignment and repeated unit identifiers, when available, stay together. A separate full-training fit predicts validation. The known randomized design remains the preferred propensity for synthetic A. Criteo estimates diagnose the released sample and cannot recover the original trial assignment probabilities.

Models and row-aligned train OOF and validation probabilities are saved locally in `artifacts/session_08/models/`. The manifest records fold membership provenance, versions, feature order, configuration and fingerprints of the ordered feature/assignment partitions. The positional IDs for Criteo are meaningful only with the unchanged ordered partition whose hash matches. Synthetic rows include unit IDs. The validation fold sentinel is −1. Use trusted Joblib artifacts in the recorded environment.

Empirical common support is the intersection of the two arm-specific min–max intervals. Quantiles and histograms show distribution shape; below 0.05 and above 0.95 indicate tails, and thresholds are configurable in the diagnostic function. Constant scores have no covariate-level resolution. A support restriction is descriptive and would change the target population if applied. No rows were removed.

Scenario C permits truth RMSE and MAE because the simulator's propensity conditions on the same observed features. F's stored truth additionally conditions on hidden U. The validation diagnostics provide no observed-feature truth for F and do not label the difference from full-information scores as prediction error. The analysis does not establish exchangeability from assignment prediction or covariate balance.

Run `python -m causallift.session08`. Use `--fixture` offline or `--output` for a separate directory. [Completion report](session-08-completion.md)
