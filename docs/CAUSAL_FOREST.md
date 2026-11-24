# Causal Forest benchmark

Session 07 uses EconML’s `CausalForestDML`. It cross-fits standardized logistic outcome probability nuisances in three training-only folds. Binary treatment uses a fixed-probability classifier with the known synthetic assignment probability or the released Criteo training share. The forest fits an orthogonalized local treatment-effect model. Honesty separates split selection from estimation within trees. The fixed profile is 100 trees, depth at most 12, leaf size at least 35, subsample fraction 0.4, two jobs and library inference enabled.

The Criteo 200k training partition has 120,000 rows. The default 1M forest cap also uses 120,000 rows, selected deterministically within training with joint assignment/outcome stratification. LinearDML and the partially linear estimator use the entire training partition. All models use the same fixed validation population for ranking and targeting. The cap bounds compute and leaves a training-size difference in the 1M comparison. Increase it explicitly with `--forest-max-train` if resources permit.

The library supplies pointwise 95% CATE intervals and its own `ate_interval(X)` for average effects. Average intervals use the library aggregation procedure. The reports use all synthetic validation rows and a deterministic 4,000-row inference sample per real validation population. Predictions and ranking metrics still use the entire validation population. Invalid conditional intervals are counted, and supported intervals determine reported mean width and synthetic pointwise coverage. Average-effect inference is for the stated supported inference rows. If the library cannot produce a finite ordered interval, the result is unavailable with a warning. Coverage on a single validation cohort describes pointwise coverage and has no simultaneous guarantee. When the library flags rank-deficient linear inference, its intervals are withheld.

Feature importance in the model manifest describes variables used by the fitted forest. It does not establish that modifying a feature causally changes responsiveness. Raw predictions are retained and the manifest reports effects outside the binary bounds. Criteo’s unknown original propensities, privacy subsampling, pooled trials and missing unit and trial IDs limit the interpretation of both effect estimates and intervals. Ranking follows Session 05’s released validation-share convention. [EconML forest API](https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html) · [Uplift conventions](UPLIFT_EVALUATION.md)

## Setup and reproduction

```powershell
.venv/Scripts/python -m pip install -e ".[dev,advanced]"
.venv/Scripts/python -m causallift.session07
.venv/Scripts/python -m causallift.session07 --fixture --output data/processed/session07_fixture
```

Use `--development-only` to skip Criteo 1M, `--forest-max-train` to set the cap, and `--repetitions` and `--seed` to configure the simulation. The run reads existing datasets and Session 04–06 artifacts. Python 3.13.15 on Windows supported EconML 0.17.1 with NumPy 2.5.5, SciPy 1.18.1, scikit-learn 1.9.1 and LightGBM 4.7.0 in the existing environment. The manifest records those and transitive package versions. Reload was verified in that same environment. Joblib compatibility across versions was not tested, so use the recorded environment for these local artifacts. [EconML package metadata](https://pypi.org/project/econml/0.17.1/) · [Completion report](session-07-completion.md)
