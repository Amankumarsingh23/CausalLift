import json

import pandas as pd
import pytest

from causallift.session07 import _training_subset, run
from causallift.simulation import simulate
from tests.test_data import FIXTURE_PATH


def test_forest_training_cap_preserves_representation_and_reproducibility():
    frame, _, spec = simulate("D", 900, 18)
    data = frame.assign(split="train")
    subset = _training_subset(data, spec, "outcome", 300, 19)
    assert len(subset) == 300
    assert set(subset["split"]) == {"train"}
    assert subset.groupby(["treatment", "outcome"]).ngroups == 4
    pd.testing.assert_frame_equal(subset, _training_subset(data, spec, "outcome", 300, 19))


def test_offline_advanced_reports_execute_with_optional_dependency(tmp_path):
    pytest.importorskip("econml")
    fixture = tmp_path / "tests/fixtures/criteo_schema.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(FIXTURE_PATH.read_bytes())
    summary = run(tmp_path, fixture=True, repetitions=2)
    assert summary["models"] == 6
    assert summary["benchmark_evaluation_partitions"] == ["validation"]
    out = tmp_path / "artifacts/session_07_fixture"
    reports = pd.read_csv(out / "final_causal_ml_comparison.csv")
    targets = pd.read_csv(out / "targeting_comparison.csv")
    assert reports.groupby(["dataset", "outcome"]).endpoint_gain.nunique().eq(1).all()
    assert targets.groupby(["dataset", "outcome", "fraction"]).n.nunique().eq(1).all()
    assert reports.loc[reports["dataset"].eq("local_fixture"), "root_pehe"].isna().all()
    manifest = json.loads((out / "causal_model_manifest.json").read_text(encoding="utf-8"))["models"]
    assert all(row["reload_verified"] and (out / row["model_path"]).is_file() for row in manifest)
    assert (out / "causal_ml_comparison.png").read_bytes().startswith(b"\x89PNG")
