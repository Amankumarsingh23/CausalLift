# CausalLift

CausalLift builds reproducible datasets for estimating incremental treatment effects and evaluating treatment allocation. Session 01 establishes the data foundation with Criteo v2.1 and six synthetic benchmarks with known conditional effects.

## Setup

Use Python 3.11 or newer. The Session 01 run used Python 3.13 on Windows.

```powershell
py -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
```

On macOS or Linux, use `python3 -m venv .venv` and replace `.venv/Scripts/python` with `.venv/bin/python`. The recorded Windows dependency versions are in [configs/requirements-session01-py313-win.txt](configs/requirements-session01-py313-win.txt).

## Build and verify Session 01

Run from the repository root. The download command fetches Criteo’s approximately 311 MB archive and checks its SHA-256. The build profiles the full archive, writes a 200,000-row stratified working sample and six 20,000-row synthetic datasets, assigns partitions, and generates the reports.

```powershell
.venv/Scripts/python -m causallift.session01 download
.venv/Scripts/python -m causallift.session01 session01
.venv/Scripts/python -m ruff check .
.venv/Scripts/python -m pytest -q
```

Use `session01 --dataset PATH --seed 42 --size 200000 --rows 20000` to set the input and defaults explicitly. Use `--synthetic-only` for an offline run. Pass `--root PATH` before the subcommand to keep separate runs in separate directories. `--size 1000000` is a larger optional working benchmark and requires more memory. See [DATASETS.md](docs/DATASETS.md) for manual download instructions. CI uses only generated data and a local fixture for test execution.

The commands write Criteo Parquet under `data/processed/`, synthetic observed and ground-truth Parquet under `data/synthetic/` and metrics under `artifacts/session_01/`. See the [data contract](docs/DATA_CONTRACT.md), [simulator](docs/CAUSAL_SIMULATOR.md) and [Session 01 completion report](docs/session-01-completion.md).

## Run Session 02

Run the experiment analysis against the existing Session 01 data.

```powershell
.venv/Scripts/python -m causallift.session02
```

Reports are written to `artifacts/session_02/`. Use `--mode synthetic` for just the synthetic benchmarks or `--mode fixture` for a small offline example that needs no Session 01 data. See the [methods and reproduction commands](docs/EXPERIMENTATION.md) and [Session 02 completion report](docs/session-02-completion.md).

## Run Session 03

Generate power and MDE plans, experiment-health diagnostics, guardrail decisions, and statistical simulations from the existing datasets and reports.

```powershell
.venv/Scripts/python -m causallift.session03
```

Reports are written to `artifacts/session_03/`. See the [methods and reproduction commands](docs/EXPERIMENT_DESIGN.md) and [Session 03 completion report](docs/session-03-completion.md).

## Run Session 04

```powershell
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m causallift.session04
```

Reports and local model files are written to `artifacts/session_04/`. Use `--development-only` to skip the optional Criteo 1M benchmark or `--fixture` for the small offline run. See the [predictive baselines guide](docs/PREDICTIVE_BASELINES.md) and [Session 04 completion report](docs/session-04-completion.md).

## Run Session 05

```powershell
.venv/Scripts/python -m causallift.session05
```

The benchmark reuses existing splits and saved predictive models. Use `--development-only` to skip Criteo 1M or `--fixture` for an offline smoke run. See the [learner guide](docs/META_LEARNERS.md), [evaluation conventions](docs/UPLIFT_EVALUATION.md) and [Session 05 results](docs/session-05-completion.md).

## Run Session 06

```powershell
.venv/Scripts/python -m causallift.session06
```

The run reuses Session 04/05 models for the original benchmarks and saves new X/R models and diagnostics under `artifacts/session_06/`. Use `--development-only` to skip Criteo 1M or `--fixture` for the standalone offline fixture. See the [cross-fitting guide](docs/CROSS_FITTING.md) and [Session 06 results](docs/session-06-completion.md).

## Run Session 07

```powershell
.venv/Scripts/python -m pip install -e ".[dev,advanced]"
.venv/Scripts/python -m causallift.session07
```

Session 07 adds partially linear DML, LinearDML and CausalForestDML. It reuses the fixed splits and saved models from earlier sessions. Use `--fixture` for an offline run or `--development-only` to skip Criteo 1M. See the [DML guide](docs/DOUBLE_MACHINE_LEARNING.md), [Causal Forest guide](docs/CAUSAL_FOREST.md) and [Session 07 results](docs/session-07-completion.md).
## Run Session 08

Run `python -m causallift.session08` in the project environment after installing the package dependencies. Use `--fixture` to run without the benchmark files. See the [DAG guide](docs/CAUSAL_DAGS.md), [propensity guide](docs/PROPENSITY_SCORES.md), and [completion report](docs/session-08-completion.md).
## Run Session 09

Run `python -m causallift.session09` to reuse the Session 01 and 08 artifacts. Use `--download` to acquire the optional NSW/CPS files or `--fixture` for an offline smoke run. See the [methods](docs/OBSERVATIONAL_ESTIMATORS.md) and [completion report](docs/session-09-completion.md).
## Run Session 10

Run `python -m causallift.session10` or use `--fixture` for an offline run. See the [causal reliability guide](docs/CAUSAL_RELIABILITY.md) and [completion report](docs/session-10-completion.md).
