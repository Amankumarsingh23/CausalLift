# Reproduce CausalLift

## Public checkout

Install with `python -m pip install -e ".[dev,advanced,platform]"` on Python 3.11 or 3.13. Omit `advanced` for the core and platform profiles without EconML. Use the [demo guide](DEMO_GUIDE.md) for native, Docker and optional MLflow commands. The checkout contains the historical aggregate artifacts and the final aggregate evidence.

```sh
python -m causallift.session13 --fixture --output artifacts/session_13_demo
python -m causallift.session14 --fixture
python -m pytest -q
python -m ruff check .
python -m pip check
```

Each Session 14 run uses a fresh output directory. Its fixture writes under `artifacts/session_14_fixture` by default and uses generated synthetic inputs and deterministic scores. It exercises the action lock, outcome join and estimator aggregation. It is separate from the real frozen-model evaluation.

## Full analytical workflow

Read the [dataset acquisition instructions](DATASETS.md) and [data contract](DATA_CONTRACT.md). From the project root, run `python -m causallift.session01 download` to acquire and verify the Criteo archive, then `python -m causallift.session01 session01` for the 200k sample and synthetic cohorts. For the 1M working sample, use `python -m causallift.session01 --root data/processed/benchmark_1m session01 --dataset data/raw/criteo-research-uplift-v2.1.csv.gz --size 1000000`. Verify the dataset's terms before acquiring or sharing it.

Run `python -m causallift.sessionNN` consecutively for sessions 02–12. Session 09 supports `--download` for the optional external NSW/CPS benchmark. Methods, switches and prerequisites are detailed below. The core DuckDB profiles read source and Parquet aggregates. The Parquet files hold observed features, split membership and separate synthetic truth. Models are fitted on train, with development comparisons on validation. Manifests record model configuration, ordered training fingerprints and the feature allowlist.

- [Experimentation](EXPERIMENTATION.md) · [Design and power](EXPERIMENT_DESIGN.md) · [Predictive baselines](PREDICTIVE_BASELINES.md)
- [Meta-learners](META_LEARNERS.md) · [Uplift evaluation](UPLIFT_EVALUATION.md) · [Cross-fitting](CROSS_FITTING.md) · [Double Machine Learning](DOUBLE_MACHINE_LEARNING.md) · [Causal Forest](CAUSAL_FOREST.md)
- [Synthetic simulator](CAUSAL_SIMULATOR.md) · [Causal DAGs](CAUSAL_DAGS.md) · [Propensity scores](PROPENSITY_SCORES.md) · [Observational estimators](OBSERVATIONAL_ESTIMATORS.md) · [Reliability](CAUSAL_RELIABILITY.md)
- [Treatment optimization](TREATMENT_OPTIMIZATION.md) · [Offline policy evaluation](OFFLINE_POLICY_EVALUATION.md) · [Business case study](BUSINESS_CASE_STUDY.md)

Do full methodology reproduction in a separate workspace to preserve the checked-in aggregate evidence. In particular, Session 12 creates a new protocol for the artifacts that are actually present. Independently retrained binaries may have different bytes.

## Replay the exact locked evaluation

Use the original local observed Parquets, synthetic truth, manifests and exact frozen Session 04/05 model binaries. From the repository root run `python -m causallift.session14` into an unused default location or specify `--output <fresh-directory>` when the published evidence already exists. The exact replay checks the protocol SHA-256 and model hashes and stops affected cohorts if provenance is missing or differs. The known hashes are recorded in the [frozen protocol](../artifacts/session_12/final_evaluation_protocol.json) and [final manifest](../artifacts/session_14/final_evaluation_manifest.json).

Before outcome access, the replay projects ordered identities and allowed features, checks published split sizes and train fingerprints, and records the action archive hashes. Criteo has no publisher row ID, so the replay uses the persisted Parquet row position within each working sample. The outcome read uses the same unchanged file and persisted test order. Nobody needs private row-level data to view the [published aggregate results](FINAL_BENCHMARK_REPORT.md).
