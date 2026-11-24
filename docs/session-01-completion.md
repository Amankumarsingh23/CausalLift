# CausalLift Session 01 completion report

## A. Implementation

Session 01 is complete. The installable package downloads or ingests Criteo, profiles the full source, samples the working dataset, simulates six scenarios, persists leakage-safe partitions, and checks DuckDB metrics against independent pandas calculations. Synthetic evaluation data lives in separate Parquet files.

## B. Repository and environment

The repository uses a `src/` package, `pyproject.toml`, a console entry point and a GitHub Actions workflow. Key modules are `data/criteo.py`, `data/contract.py`, `data/splits.py`, `data/analytics.py`, `simulation/__init__.py`, and `session01.py`, all under `src/causallift/`.

The verified environment was Python 3.13.15 on Windows with NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1, scikit-learn 1.9.1, PyArrow 26.0.1, DuckDB 2.0.1, Matplotlib 3.11.2, pytest 9.2.0 and Ruff 0.17.6. The package declares Python 3.11+ and the workflow covers 3.11 and 3.13. See [recorded environment](../configs/requirements-session01-py313-win.txt).

## C. Real dataset

Criteo Uplift Prediction Dataset v2.1 was downloaded from Criteo’s official mirror and verified against its published SHA-256. The original link failed DNS resolution at its Azure Blob redirect. The full 13,979,592 rows were read and checked. The schema contains 12 features and four binary fields. Zero missing or invalid values were found in the archive or working sample. The default output is a 200,000-row Parquet sample stratified by assignment and conversion with seed 42. [Criteo AI Lab release](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) · [Publisher archive](https://huggingface.co/datasets/criteo/criteo-uplift/blob/main/criteo-research-uplift-v2.1.csv.gz).

| Data | Arm | Rows | Conversion | Visit |
| --- | --- | ---: | ---: | ---: |
| Full archive | Control | 2,096,937 | 0.1938% | 3.8201% |
| Full archive | Treated | 11,882,655 | 0.3089% | 4.8543% |
| Working sample | Control | 30,000 | 0.1933% | 3.8100% |
| Working sample | Treated | 170,000 | 0.3088% | 4.8171% |

The optional one-million-row benchmark also built locally. It has 150,000 control rows and 850,000 treated rows, with 600,000/200,000/200,000 train, validation and test rows. Control and treated conversion rates are 0.1940% and 0.3089%, and visit rates are 3.7200% and 4.8609%. Its profile is under `data/processed/benchmark_1m/artifacts/session_01/`.

## D. Synthetic datasets

Each scenario uses seed 42 and 20,000 rows. Effects are in percentage points. F reports full-information CATE and ATT in this table.

| Scenario | Rows | Treated | Outcome prevalence | True ATE | True ATT | Naive difference | True CATE range |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A. Constant | 20,000 | 49.585% | 39.675% | +10.000 | +10.000 | +10.389 | +10.00 to +10.00 |
| B. Nonlinear | 20,000 | 49.585% | 34.760% | −0.066 | −0.128 | +0.477 | −19.00 to +18.97 |
| C. Observed confounding | 20,000 | 44.025% | 36.295% | −0.066 | +4.213 | +14.776 | −19.00 to +18.97 |
| D. Imbalanced | 20,000 | 15.300% | 34.325% | −0.066 | −0.136 | −0.438 | −19.00 to +18.97 |
| E. Mixed signs | 20,000 | 49.585% | 35.875% | +1.888 | +1.900 | +2.726 | −12.00 to +16.00 |
| F. Hidden confounding | 20,000 | 49.985% | 43.700% | −0.057 | +5.907 | +19.806 | −24.30 to +24.75 |

## E. Partitions and leakage

| Dataset | Train | Validation | Test | Treated across partitions | Outcome across partitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Criteo working sample | 120,000 | 40,001 | 39,999 | 84.998–85.002% | 0.2900–0.2925% |
| A | 12,000 | 4,001 | 3,999 | 49.583–49.588% | 39.665–39.685% |
| B | 12,001 | 4,000 | 3,999 | 49.575–49.588% | 34.750–34.764% |
| C | 12,000 | 4,000 | 4,000 | 44.025% | 36.292–36.300% |
| D | 12,000 | 4,000 | 4,000 | 15.300% | 34.325% |
| E | 12,000 | 4,001 | 3,999 | 49.583–49.588% | 35.866–35.884% |
| F | 12,000 | 4,001 | 3,999 | 49.983–49.988% | 43.689–43.711% |

The split column assigns exactly one partition to each row. Every partition contains both arms. The repeated-ID fixture confirms no unit crosses partitions. Criteo supplies no customer ID, so coverage is verified at the observation level. Tests verify that treatment, outcomes, exposure, IDs, splits, ground truth and latent values stay outside the model feature allowlist. No fitted preprocessing was applied.

## F. Verification

| Local command | Result |
| --- | --- |
| `.venv/Scripts/python -m pip install --no-build-isolation -e '.[dev]'` | Package installed |
| `.venv/Scripts/python -m causallift.session01 download` | Existing archive verified against Criteo SHA-256 |
| `.venv/Scripts/python -m causallift.session01 session01` | Real and synthetic datasets, metadata and reports generated |
| `.venv/Scripts/python -m causallift.session01 --root data/processed/benchmark_1m session01 --dataset data/raw/criteo-research-uplift-v2.1.csv.gz --size 1000000` | Optional benchmark built |
| `.venv/Scripts/python -m ruff check .` | All checks passed |
| `.venv/Scripts/python -m pytest -q` | 34 passed, zero failures |
| `.venv/Scripts/python -m pip check` | No broken requirements |

The initial tests caught a probability bound error in scenario F. The corrected baseline guarantees valid probabilities and the full build succeeded on rerun. GitHub Actions is configured in `.github/workflows/ci.yml`. No hosted run was initiated.

## G. Generated artifacts

- [artifacts/session_01/dataset_summary.json](../artifacts/session_01/dataset_summary.json)
- [artifacts/session_01/data_quality_report.json](../artifacts/session_01/data_quality_report.json)
- [artifacts/session_01/synthetic_scenario_summary.csv](../artifacts/session_01/synthetic_scenario_summary.csv)
- [artifacts/session_01/split_summary.csv](../artifacts/session_01/split_summary.csv)
- [artifacts/session_01/synthetic_effect_distribution.png](../artifacts/session_01/synthetic_effect_distribution.png)
- [artifacts/session_01/propensity_overlap.png](../artifacts/session_01/propensity_overlap.png)

## H. Limitations and blockers

Real-data acquisition succeeded. Criteo’s privacy subsampling limits interpretation to the released benchmark. The archive omits trial and user identifiers and does not supply the exact assignment probabilities or counterfactual outcomes. Scenario F deliberately violates exchangeability given the observed covariates. Grouped splits can deviate from target size and prevalence based on observed group structure.

## I. Session 2 handoff

The ready-to-use real dataset is `data/processed/criteo_v2_1.parquet`. Synthetic datasets are `data/synthetic/synthetic_{A..F}.parquet`, with evaluation truth in `synthetic_{A..F}_truth.parquet`. Each observed file has a sibling `.spec.json`.

```python
from pathlib import Path

from causallift.data.analytics import parquet_summary
from causallift.data.contract import load_dataset
from causallift.data.criteo import CRITEO

path = Path("data/processed/criteo_v2_1.parquet")
train = load_dataset(path, CRITEO, "train")
X_train = CRITEO.features(train)
print(parquet_summary(path, CRITEO, by_split=True))
```

Use the [README commands](../README.md) to rebuild the input data. The Session 2 analysis engine is pending.
