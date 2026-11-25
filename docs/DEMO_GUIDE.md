# Local demo guide

Run from the repository root. The demo uses published aggregates and a seeded fixture. It needs no Criteo or NSW/CPS downloads, model training, or API credentials.

## Native setup

```powershell
python -m pip install -e ".[dev,platform]"
python -m uvicorn causallift.platform.api:app --port 8000
```

In a second terminal:

```powershell
$env:CAUSALLIFT_API = "http://127.0.0.1:8000"
python -m streamlit run dashboard/app.py
```

Open `http://127.0.0.1:8501`. The API explorer is at `http://127.0.0.1:8000/docs`. Leaving the API variable unset runs interactive requests through the same services directly.

1. In **Experimentation lab**, analyze 120/1,000 versus 100/1,000. The result is +2.00 percentage points. Plan a two-sided experiment at baseline 10%, absolute MDE 2 percentage points and 80% power. It requires 3,835 per arm.
2. In **Uplift comparison**, choose a cohort and outcome. Session 05 plots stored Qini coordinates. The later sessions expose published comparisons and metric tables.
3. Review the adjustment sets and distinct ATE and ATT targets in **Observational inference**. Inspect Scenario F in **Causal reliability**.
4. In **Business policy lab**, change the profit-aware fixture budget from 120 to 60 with cost 2 and contribution 45. The selection falls from 60 to 30 with a modeled profit of 114.22 at the lower budget. Try the treat-none policy or raise treatment cost to 10.
5. In **Reports**, download the aggregate CSV and executive Markdown. The Criteo report uses hypothetical currency and paired fixed-action uncertainty.

## Docker

```sh
docker compose up --build
```

Compose runs the API and dashboard with read-only benchmark artifacts, internal service communication, and health checks. It publishes ports 8000 and 8501 on localhost. Local model availability is optional and never required for optimization. To add local row-data availability, use a Compose override with a read-only `./data/processed:/app/data/processed:ro` mount on both services. Leave it out for the demo. Stop the containers with `docker compose down`.

## Verification and optional MLflow

```powershell
python -m causallift.session13 --fixture --output artifacts/session_13
python -m pytest -q tests/test_platform.py
python -m pip install -e ".[tracking]"
$env:MLFLOW_ALLOW_FILE_STORE = "true"
python -m causallift.session13 --fixture --output artifacts/session_13 --tracking-dir .mlflow
```

The MLflow filesystem store is local and ignored by Git. Recent MLflow versions require the explicit filesystem-store opt-in shown above. Bash users can use `export` for the environment variables. Use a separate `--output` directory to preserve the checked-in evidence.

## Four-minute interview walkthrough

1. Open **Overview** and identify the published validation, reserved-test and fixture designations.
2. In **Experimentation lab**, analyze the default A/B counts and explain the outcome-rate difference and its interval.
3. Compare predictive conversion rank with incremental-effect rank in **Uplift comparison**. Select a cohort and show the published ranking metrics.
4. Open Scenario F in **Causal reliability** and explain why stable estimates can remain biased when a confounder is hidden.
5. In **Business policy lab**, lower the budget to 60 and show the change in selected customers and planned spending.
6. In **Reports**, show the paired DR comparisons and spend differences. Synthetic E versus predictive is inconclusive. Criteo 1M versus predictive is positive under hypothetical economics, and its interval against treat-none crosses zero.
7. Download the executive brief with both validation and test results. [Final browser evidence](../artifacts/session_14/browser_smoke_results.json)
