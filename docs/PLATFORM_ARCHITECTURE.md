# Decision platform architecture

Streamlit presents seven views over the published development artifacts. The dashboard uses the same service functions as FastAPI in native standalone mode. Set `CAUSALLIFT_API` to route interactive requests to the API, as Compose does. Benchmark display comes from the shared artifact registry in both modes.

| Layer | Responsibility |
| --- | --- |
| Artifact registry | Allowlisted aggregate tables, manifests and plots. Cached by file metadata. Reports missing assets explicitly. |
| Shared services | Validate bounded inputs and delegate to existing experiment, planning and allocation modules. |
| FastAPI | Versioned local endpoints, structured request and artifact errors, and a downloadable decision brief. |
| Streamlit | Interactive experiment and policy forms, Plotly charts from stored values, diagnostic tables, CSV and Markdown exports. |
| Optional tracking | MLflow records fixture parameters, seed, fingerprints and summary metrics to a local store. |

The API exposes `/health`, `/api/v1/catalog`, `/api/v1/overview`, `/api/v1/reliability`, and `/api/v1/{group}/benchmarks`. Benchmark queries accept an allowlisted `artifact` ID, exact dataset, outcome and model filters where supported, and a maximum of 500 rows per request. POST routes are `/api/v1/experiments/analyze`, `/api/v1/experiments/plan`, and `/api/v1/policies/optimize`. The report is served at `/api/v1/reports/business`. Open `/docs` on the local API for the request schemas.

Artifact keys are fixed relative identifiers. The registry rejects path aliases and uses resolved paths inside approved directories. Client-supplied paths, model objects and arbitrary code have no interface. Full-local asset detection checks availability only. Interactive policy decisions always use the same reproducible fixture. The fixture has 240 synthetic customers, six bounded decision inputs, and deterministic scores from pre-treatment engagement and loyalty. Scenario E truth is constructed after allocation and used only for aggregate scoring. Scenario F illustrates unsupported identification and suppresses its expected-truth comparison.

The reserved test partition stays locked. The application consumes earlier published aggregates and never loads reserved evaluation rows. Session 12's frozen file and model artifacts remain untouched. Scenario F stays research-only, and the DML coverage warning is carried forward.

The application binds to loopback in native execution and publishes Compose ports on loopback. Authentication and public hosting are outside its scope. The regular package runs without the `platform` or `tracking` extras.
