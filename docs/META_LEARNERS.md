# S-, T-, X- and R-learners

Session 05 fits four binary meta-learners using the existing training split. The S-learner fits one model on allowed pre-treatment features plus a controlled binary assignment field. Inference toggles assignment on otherwise identical rows. The T-learner fits one model on each assigned arm using the same feature allowlist. Both return control probability `mu0`, treated probability `mu1` and effect `mu1 − mu0`.

Logistic regression uses L2 regularization and scales pre-treatment features within the training fit. S leaves the 0/1 assignment column unscaled and includes no explicit treatment-by-feature interactions. Its additive log-odds coefficient has a fixed effect sign on the probability scale, while the magnitude can vary with baseline risk. T fits separate coefficients that can change the effect sign. LightGBM uses a fixed deterministic binary configuration of 100 trees, up to 15 leaves per tree, and no early stopping or search. S can share information across arms but may underuse treatment. T can model distinct relationships but may have higher variance in small or sparse arms. A single-class fit uses the observed constant 0 or 1 probability and records the fallback.

The benchmark applies the dataset contract before fitting. Ordinary inputs exclude assignment, exposure, outcomes, identifiers and truth. The estimator checks finite numeric features and the fitted feature order. Validation outcomes are used for benchmarking only by the Session 05 models. No test rows enter model fitting or evaluation. Session 04 predictive baselines are loaded from their saved artifacts and checked against the training fingerprint. Their original validation-based early stopping remains part of their provenance.

Identification relies on randomized assignment or exchangeability, consistency, positivity and no interference, together with suitable outcome models. Missing original trial probabilities and privacy subsampling limit Criteo’s interpretation. The randomized synthetic benchmark separates observed fit data from conditional-effect truth. [Uplift evaluation](UPLIFT_EVALUATION.md)

```python
from causallift.meta_learners import MetaLearner

model = MetaLearner("t", "logistic").fit(X_train, treatment_train, outcome_train)
potential = model.predict_potential_outcomes(X_validation)
effect = potential.effect
```

For callers that only need effects, `predict_effect(X)` is the common interface. Local Joblib files store the fitted models, and the manifest records ordered features, training fingerprints including assignment, outcome, seed, configuration, environment, timings, arm counts and validation metrics. The run reloads every artifact and compares both potential-outcome predictions on validation.

## X- and R-learners

Session 06 adds two bases for each learner. Linear uses logistic probability estimation and standardized Ridge effect regression. LightGBM uses a binary outcome model and a squared-error effect regressor with fixed configurations of 65 trees and up to 15 leaves per tree. Session 05 S/T artifacts retain their original configurations. The small repeated-seed study fits T-LightGBM with its original 100-tree limit and X/R-LightGBM with 65 trees.

X fits the two arm-specific probability models used by T-learning. It imputes `D0 = mu1(X) − Y` for controls and `D1 = Y − mu0(X)` for treated rows, using the opposite arm’s model. It fits separate effect regressions and combines them as `e(X) × tau0(X) + (1 − e(X)) × tau1(X)`. When few units are treated, this places more weight on the pseudo-effect model that uses the larger arm’s control outcome model for imputation. Benefits depend on the structure of the effects and both stages’ errors. Single-class arm models use the existing constant-probability fallback. [X-learner paper](https://pubmed.ncbi.nlm.nih.gov/30770453/)

R fits cross-fitted outcome nuisance models and regresses outcome residuals on treatment residuals through an effect function. Both X and R use the known design propensity in randomized synthetic data or Criteo’s released training assignment share. They implement `predict_effect(X)` and report unconstrained regression predictions, including counts outside the feasible binary CATE range. No clipping changes ranking or accuracy. [Cross-fitting and weighted loss](CROSS_FITTING.md)
