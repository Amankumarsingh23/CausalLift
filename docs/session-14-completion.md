# Session 14 completion

The original frozen benchmark completed for synthetic E, A, B and D and Criteo 1M and 200k. The locked evaluation took 7.87 seconds in the original Windows Python 3.13 run. It checked all 18 model binaries used by the frozen replay against the published SHA-256 values, train fingerprints and predictor allowlists. No models were fitted or selected after test access. [Final evaluation manifest](../artifacts/session_14/final_evaluation_manifest.json)

The Session 12 protocol remained byte-identical on the execution checkout. Its SHA-256 before and after was `35cf618174130b6c527112aeda1b530fbcb460f0ddbadca0ed56f0b264f1f13e`. The LF Git checkout has a separately verified frozen hash. Session 12's `test_evaluated: false` is the historical freeze state. [Protocol verification](../artifacts/session_14/frozen_protocol_verification.json)

The execution projected ordered row identity, split and allowed pre-treatment features before reading evaluation labels. It checked membership counts against published Session 01 metadata and model training inputs against manifests. The allocation lock was written at 13:50:02.236884 UTC and the outcome-access marker at 13:50:02.256228 UTC. The ordered action vector digests, local archive digests and observed dataset digests are recorded per cohort. Criteo row identity is the ordinal in its original persisted working sample. After locking, the replay checked file identity and joined labels in the original test order. [Action freeze](../artifacts/session_14/action_freeze.json) · [Outcome access marker](../artifacts/session_14/outcome_access.json)

## Findings

At the 20% setting, synthetic E profit-aware versus predictive was +286.82 by paired DR with a conditional 95% interval of [−2,883.03, 3,456.67]. The known expected difference was +2,179.80. Against treat-none, DR profit was +3,583.05 and true expected profit was +4,142.20. On Criteo 1M, the paired difference versus predictive was +11,117.61 hypothetical units, with a conditional interval of [4,858.33, 17,376.90]. The profit-aware policy spent less and had 32.05 fewer DR incremental conversions than predictive. Its interval against treat-none spanned zero.

The [consolidated benchmark report](FINAL_BENCHMARK_REPORT.md) includes secondary results, per-unit validation comparisons and earlier unfavorable findings. Criteo trial-level propensities and clustering are unavailable. Criteo economics are hypothetical, and the evaluation samples overlap. Fixed-action intervals omit allocation reoptimization, nuisance fitting and unknown trial clustering. Scenario F remains research-only.

## Release evidence

The [reproduction guide](REPRODUCTION.md) separates methodology reproduction from exact replay. The [resume evidence](RESUME_EVIDENCE.md) maps measured capabilities to code, tests and artifacts. The [demo guide](DEMO_GUIDE.md) documents the artifact-only workflow. All seven dashboard pages rendered in Edge, including the final aggregate display. The browser verified interactive calculations and actual exports. [Browser verification](../artifacts/session_14/browser_smoke_results.json) · [Final verification](../artifacts/session_14/final_verification.json)

The repository publishes compact aggregates and provenance. It excludes raw and processed data, model binaries, local action vectors, credentials and local tracking stores.
