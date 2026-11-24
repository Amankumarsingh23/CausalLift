import json

import joblib
import pandas as pd

from causallift.session04 import run
from tests.test_data import FIXTURE_PATH


def test_predictive_reports_offline_and_model_roundtrip(tmp_path):
    root = tmp_path / "inputs"
    fixture = root / "tests/fixtures/criteo_schema.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(FIXTURE_PATH.read_bytes())
    target = tmp_path / "output"
    summary = run(root, fixture=True, output=target)
    assert summary["models"] == 9
    assert summary["evaluated_partitions"] == ["validation"]
    assert set(row["dataset"] for row in summary["real_results"]) == {"local_fixture"}
    assert len(pd.read_csv(target / "targeting_baseline_results.csv")) == 21
    manifest = json.loads((target / "predictive_model_manifest.json").read_text(encoding="utf-8"))["models"]
    for entry in manifest:
        assert entry["split_provenance"]["training"] == "train"
        assert entry["split_provenance"]["evaluation"] == "validation"
        assert len(entry["training_fingerprint_sha256"]) == 64
        assert set(entry["feature_allowlist"]).isdisjoint({"treatment", "exposure", "conversion",
                                                          "outcome", "split", "unit_id", "tau_true"})
        model = joblib.load(target / entry["model_path"])
        assert model.name == entry["model"]
    for filename in ("precision_recall_curves.png", "roc_curves.png", "calibration_curves.png", "targeting_comparison.png"):
        assert (target / filename).read_bytes().startswith(b"\x89PNG")
