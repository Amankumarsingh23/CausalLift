# Predictive baselines and targeting

## Data and model choices

The benchmark reads the existing Session 01 train and validation partitions of Criteo 200k, the optional Criteo 1M sample, and randomized synthetic scenarios A, B, D and E. Conversion is the primary Criteo target and visit is secondary. Criteo has about 0.29% conversion prevalence. Models fit the original class distribution with no resampling or class reweighting. The two Criteo samples overlap, so their validation results are descriptive comparisons.

The constant baseline predicts the training prevalence. Logistic regression uses an L2-regularized sklearn pipeline with a standard scaler fitted on train. LightGBM uses a fixed binary objective configuration with at most 160 trees, 15 leaves per tree and 15 rounds of validation log-loss patience. Its validation metrics are development benchmarks because validation also determines when training stops. The final test partition is reserved. [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)

All predictors are selected through the existing pre-treatment feature allowlist. The model interface selects the same ordered columns at fit and prediction. Treatment assignment, exposure, outcomes, identifiers, split labels, latent variables and potential-outcome truth are outside the predictor set. Predictions estimate the observed outcome probability under the logged assignment mixture. Recovering either assignment-specific potential-outcome mean or a conditional effect requires a different estimand and training design.

## Validation and calibration

ROC-AUC describes ranking. Average precision weights precision by increases in recall and is the primary PR summary, distinct from trapezoidal PR area. The no-skill precision reference is the observed prevalence. Log loss and Brier score evaluate probability forecasts. Single-class validation results retain probability scores and flag ranking metrics as unavailable.

Reliability bins use up to ten quantile bins and keep tied scores together. Empty bins are omitted. The CSV records each bin's count, positives, score range, average prediction and observed rate. Constant predictions occupy one bin. The figures scale to observed bin means. Calibration is assessed on validation without fitting a separate calibrator.

## Equal-capacity targeting

One prespecified policy ranks by the LightGBM conversion score. The constant-effect policy estimates the training assigned-arm outcome difference and gives every eligible customer the same score. All policies select `floor(fraction × validation_rows)` at 10%, 20% and 50%. Ties use a seeded common permutation. This gives random and constant-effect policies identical selections and isolates the effect of ranking.

Synthetic truth is loaded after fitting and matched by unit ID in the separate evaluation procedure. The oracle ranks by true CATE and serves as an evaluation reference. Expected incremental conversions are the sum of true CATE among selected units. This is a model-based expectation under the simulation, and can be fractional or negative. Criteo outputs contain selection counts, predictive score summaries, logged conversion rate and treatment share. Blank true effects indicate unavailable counterfactuals. Comparisons use equal capacities and no costs.

## Reproduce and load

```powershell
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m causallift.session04
.venv/Scripts/python -m causallift.session04 --fixture --output data/processed/session04_fixture
```

Use `--development-only` to skip the optional 1M input, `--seed` to set model and ranking seeds, and `--root` and `--output` to select inputs and outputs. The fixture combines the small repository CSV with a generated synthetic E scenario. Benchmark mode needs the Session 01 Parquet files already on disk.

The manifest records model files, configuration, ordered training feature/target fingerprint, provenance, versions, fit duration and metrics. A local model can be restored with `joblib.load(path)`; call `model.predict(validation_frame)` for probabilities in row order. Use local trusted artifacts with the recorded environment. Model binaries are gitignored. [Measured results and artifact index](session-04-completion.md)
