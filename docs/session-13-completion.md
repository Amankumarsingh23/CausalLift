# Session 13 completion report

FastAPI, Streamlit and Plotly expose the existing experiment, causal inference, reliability and policy analysis through a local decision platform. The offline run used 57 published assets and a deterministic 240-unit fixture. The fixture uses the existing allocation engine and does not train or load historical models. See the [architecture](PLATFORM_ARCHITECTURE.md) and [demo guide](DEMO_GUIDE.md).

## Measured verification

- Edge rendered all seven views with the native HTTP API and Streamlit, using a temporary artifact-only root. The browser ran binary analysis, sample-size planning and a budget change, then downloaded and validated both the Markdown brief and aggregate CSV. [Browser evidence](../artifacts/session_13/browser_smoke_results.json) · [Overview screenshot](../artifacts/session_13/dashboard-overview.png) · [Policy lab screenshot](../artifacts/session_13/dashboard-business-policy.png)
- The binary case returned +2.00 percentage points. The planner returned 3,835 per arm and 80.004% achieved power. Budgets of 60 and 120 selected 30 and 60 customers for expected fixture profit of 114.22 and 186.06 respectively. API checks also confirmed structured missing-artifact handling. [Fixture results](../artifacts/session_13/demo_run_manifest.json) · [API results](../artifacts/session_13/api_smoke_results.json)
- On Windows 11 and Python 3.13.15, median in-process API timings over three calls were 5.16 ms for binary analysis, 6.70 ms for planning and 4.98 ms for the budget-120 optimization endpoint. The cold Criteo benchmark CSV parse took 25.57 ms. Native service readiness was measured by polling each health endpoint from process launch. [Response times](../artifacts/session_13/response_time_summary.csv) · [Startup timings](../artifacts/session_13/browser_smoke_results.json)
- MLflow recorded an executed fixture optimization run with seed, inputs, fingerprints and aggregate metrics. The run identifier and exact local platform package versions are recorded in the verification artifact. [Platform verification](../artifacts/session_13/platform_verification.json)
- Docker Compose built and ran both services. Health checks passed and a live API request at budget 60 selected 30 fixture customers. [Docker verification](../artifacts/session_13/docker_verification.json)
- Local pytest passed 149 tests. Ruff and pip check passed. The repository workflow keeps the analytical matrix and runs isolated fixture and platform checks on both supported Python versions.

## Validation findings and handoff

Synthetic E profit-aware targeting exceeded predictive targeting by 2,230 in known expected profit. The paired DR estimate was 1,571 and its conditional interval included zero. Criteo 1M showed a paired DR difference of 7,092 under hypothetical economics with different proposed spending, and its profit-aware interval against treat-none included zero. The original trial propensities remain unavailable. Only 58 control conversions inform the Criteo 1M validation cohort. Conditional intervals omit allocation reoptimization, nuisance fitting and unknown clustering. Scenario F remains research-only. Session 07 DML coverage remains 0/30.

The frozen Session 14 protocol SHA-256 was `35cf618174130b6c527112aeda1b530fbcb460f0ddbadca0ed56f0b264f1f13e` before and after verification. The final test partition remained unused. Session 14 can run the locked protocol and reuse the platform evidence for the final demonstration.
