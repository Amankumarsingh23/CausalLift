import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causallift.data.criteo import checksum
from causallift.session14 import economics, load_frozen, run, verify_protocol
from causallift.simulation import simulate

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_economics_and_protocol_integrity():
    protocol, digest = verify_protocol(ROOT)
    assert (
        checksum(ROOT / "artifacts/session_12/final_evaluation_protocol.json") == digest
    )
    value, cost, eligibility, budgets = economics(protocol, "synthetic_E", 3999)
    assert budgets[-1] == (0.2, 1599, 1199)
    assert np.all(value == 45) and np.all(cost == 2)
    assert np.array_equal(eligibility, economics(protocol, "synthetic_E", 3999)[2])
    assert 0 < eligibility.sum() < len(eligibility)
    value, cost, eligibility, budgets = economics(protocol, "Criteo 1M", 200001)
    assert budgets[1] == (0.2, 16000, 40000)
    assert (value.min(), value.max()) == (60, 140)
    assert eligibility.all() and np.all(cost == 0.4)


def test_fixture_locks_actions_before_outcomes_and_does_not_reuse_output(tmp_path):
    summary = run(ROOT, tmp_path, fixture=True)
    lock = json.loads((tmp_path / "action_freeze.json").read_text(encoding="utf-8"))
    access = json.loads((tmp_path / "outcome_access.json").read_text(encoding="utf-8"))
    assert summary["split"] == "fixture" and not summary["test_evaluated"]
    assert lock["locked_utc"] <= access["started_utc"]
    assert checksum(tmp_path / "action_freeze.json") == access["action_freeze_sha256"]
    e = lock["cohorts"]["synthetic_E"]
    assert {"outcome", "treatment", "tau_true"}.isdisjoint(e["test_feature_columns"])
    assert all(
        row["spend"] <= row["budget"] and row["selected"] <= row["capacity"]
        for row in e["action_vectors"]
    )
    scores = pd.read_csv(tmp_path / "final_policy_evaluation.csv")
    paired = pd.read_csv(tmp_path / "final_paired_policy_comparisons.csv")
    assert set(scores.split) == {"fixture"}
    assert scores.loc[scores.policy.eq("none"), "incremental_rate"].eq(0).all()
    assert set(paired.reference) == {"none", "predictive"}
    assert paired.loc[paired.method.eq("snips"), "incremental_profit"].isna().all()
    with pytest.raises(FileExistsError, match="Preserve"):
        run(ROOT, tmp_path, fixture=True)


def test_missing_or_changed_frozen_model_fails_before_deserialization(tmp_path):
    frame, _, spec = simulate("E", 80, 42)
    dataset, outcome = "synthetic_E", spec.outcome_column
    entry = {"model_path": "models/placeholder.joblib"}
    entries = (
        {(dataset, outcome, "lightgbm"): entry},
        {
            (dataset, outcome, "s_lightgbm"): entry,
            (dataset, outcome, "t_logistic"): entry,
        },
    )
    key = "session_04/models/placeholder.joblib"
    protocol = {"model_artifact_sha256": {key: "0" * 64}}
    with pytest.raises(FileNotFoundError, match="Missing frozen model"):
        load_frozen(tmp_path, protocol, entries, dataset, spec, frame)
    path = tmp_path / "artifacts" / key
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="expected .* got"):
        load_frozen(tmp_path, protocol, entries, dataset, spec, frame)
