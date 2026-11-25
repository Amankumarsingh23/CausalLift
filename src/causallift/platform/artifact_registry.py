"""Resolve only explicitly approved aggregates and inspect local asset availability."""

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

PROTOCOL = "artifacts/session_12/final_evaluation_protocol.json"
# Git can materialize the unchanged Session 12 blob with CRLF on Windows.
FROZEN_HASHES = {
    "4e84c3bacc1ddb2858cff1715e42697fd16fd03daccf172cccaf5bd25871dfc6",
    "35cf618174130b6c527112aeda1b530fbcb460f0ddbadca0ed56f0b264f1f13e",
}
GROUPS = {
    "overview": {
        1: "dataset_summary.json split_summary.csv synthetic_scenario_summary.csv"
    },
    "experiments": {
        2: "synthetic_experiment_results.csv criteo_experiment_results.csv",
        3: "cuped_results.json monte_carlo_results.csv multiple_testing_results.csv srm_results.json guardrail_decisions.json",
    },
    "uplift": {
        4: "predictive_model_results.csv synthetic_predictive_results.csv",
        5: "causal_estimator_results.csv uplift_ranking_metrics.csv synthetic_effect_accuracy.csv uplift_curve_coordinates.csv uplift_calibration.csv",
        6: "meta_learner_comparison.csv uplift_ranking_metrics.csv synthetic_effect_accuracy.csv uplift_calibration.csv",
        7: "final_causal_ml_comparison.csv uplift_ranking_metrics.csv synthetic_effect_accuracy.csv uplift_calibration.csv dml_diagnostics.json causal_ml_comparison.png",
    },
    "observational": {
        8: "identification_results.json causal_graphs.png propensity_overlap.png propensity_overlap_summary.csv",
        9: "observational_ate_results.csv observational_att_results.csv matching_diagnostics.csv weight_diagnostics.csv covariate_balance_comparison.csv lalonde_benchmark.csv",
        10: "matching_tie_correction.csv",
    },
    "reliability": {
        10: "causal_reliability_report.json overlap_sensitivity.csv placebo_refutation.csv random_common_cause.csv subset_stability.csv negative_control_results.csv hidden_confounding_sensitivity.csv",
        7: "dml_diagnostics.json",
    },
    "policies": {11: "policy_optimization_results.csv economic_scenarios.json"},
    "evaluation": {
        12: "synthetic_policy_evaluation.csv criteo_policy_evaluation.csv paired_policy_comparisons.csv policy_support_diagnostics.csv budget_profit_frontier.csv cost_sensitivity.csv"
    },
    "model_metadata": {
        4: "predictive_model_manifest.json",
        5: "causal_model_manifest.json",
        6: "causal_model_manifest.json",
        7: "causal_model_manifest.json",
    },
    "final": {
        14: "session_14_summary.json final_business_summary.json validation_vs_test.csv final_paired_policy_comparisons.csv final_policy_evaluation.csv final_policy_support.csv"
    },
}
METRICS = {
    "absolute_lift": "Treatment minus control outcome rate",
    "qini": "Area of the Qini curve above random targeting",
    "auuc": "Area under the cumulative uplift curve",
    "root_pehe": "Root mean squared individual effect error on synthetic data",
    "ess": "Effective sample size from weights",
    "incremental_profit": "Incremental contribution minus difference in proposed spending",
    "roi": "Incremental profit divided by proposed spending",
}
PATHS = {
    f"{session:02d}/{filename}": (group, session)
    for group, sessions in GROUPS.items()
    for session, names in sessions.items()
    for filename in names.split()
}


@lru_cache(maxsize=160)
def _read(path, stamp):
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))


class ArtifactRegistry:
    def __init__(self, root=None):
        self.root = Path(
            root if root is not None else os.environ.get("CAUSALLIFT_ROOT", Path.cwd())
        ).resolve()

    def _approved(self, relative, base):
        expected = self.root / relative
        path = expected.resolve()
        if (
            path != expected
            or not path.is_relative_to(self.root / base)
            or not path.is_file()
        ):
            raise FileNotFoundError("Approved asset is unavailable")
        return path

    def path(self, key):
        if key not in PATHS:
            raise KeyError("Unknown artifact")
        session, name = key.split("/")
        base = f"artifacts/session_{session}"
        return self._approved(f"{base}/{name}", base)

    def load(self, key):
        path = self.path(key)
        return _read(path, (path.stat().st_mtime_ns, path.stat().st_size))

    def model_catalog(self):
        models = []
        try:
            protocol = self._approved(PROTOCOL, "artifacts/session_12")
            fingerprints = _read(protocol, protocol.stat().st_mtime_ns).get(
                "model_artifact_sha256", {}
            )
        except FileNotFoundError:
            fingerprints = {}
        for session, filename in GROUPS["model_metadata"].items():
            base = f"artifacts/session_{session:02d}"
            try:
                manifest = self._approved(f"{base}/{filename}", base)
            except FileNotFoundError:
                continue
            for row in _read(manifest, manifest.stat().st_mtime_ns).get("models", []):
                name = Path(row.get("model_path") or "").name
                if not name.endswith(".joblib"):
                    name = "unavailable.joblib"
                try:
                    self._approved(f"{base}/models/{name}", f"{base}/models")
                    status = "local_model"
                except FileNotFoundError:
                    status = "missing"
                models.append(
                    {
                        "session": session,
                        "dataset": row.get("dataset"),
                        "outcome": row.get("outcome"),
                        "model": row.get("policy", row.get("model")),
                        "status": status,
                        "split": "validation",
                        "training_fingerprint": row.get("training_fingerprint_sha256"),
                        "model_fingerprint": fingerprints.get(
                            f"session_{session:02d}/models/{name}"
                        ),
                        "reliability": "See the corresponding development evaluation",
                    }
                )
        return models

    def catalog(self):
        records = []
        for key, (group, session) in PATHS.items():
            try:
                self.path(key)
                status = "published_aggregate"
            except FileNotFoundError:
                status = "missing"
            records.append(
                {
                    "id": key,
                    "session": session,
                    "group": group,
                    "type": Path(key).suffix[1:],
                    "split": "reserved_test aggregate"
                    if session == 14
                    else "historical development",
                    "status": status,
                }
            )
        models = self.model_catalog()
        processed = self.root / "data/processed"
        row_data = any(
            p.is_file() and p.resolve() == p
            for p in processed.glob("*/criteo_v2_1.parquet")
        )
        local = row_data and any(row["status"] == "local_model" for row in models)
        return {
            "mode": "full-local assets detected" if local else "artifact-only",
            "interactive_policy": "synthetic fixture",
            "local_row_data": row_data,
            "artifacts": records,
            "models": models,
            "metrics": METRICS,
            "reserved_test": "locked",
        }

    def protocol_checksum(self):
        path = self._approved(PROTOCOL, "artifacts/session_12")
        return hashlib.sha256(path.read_bytes()).hexdigest()
