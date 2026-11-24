import json

import pandas as pd

from causallift.session06 import run
from tests.test_data import FIXTURE_PATH


def test_offline_combined_benchmark_and_crossfit_reports(tmp_path):
    fixture = tmp_path / "tests/fixtures/criteo_schema.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(FIXTURE_PATH.read_bytes())
    summary = run(tmp_path, fixture=True)
    assert summary["models"] == 12
    assert summary["evaluated_partitions"] == ["validation"]
    out = tmp_path / "artifacts/session_06_fixture"
    diag = json.loads((out / "crossfit_diagnostics.json").read_text(encoding="utf-8"))["runs"]
    assert len(diag) == 6
    for row in diag.values():
        assert row["covered_rows"] == row["training_n"] == sum(f["held_out"]["n"] for f in row["folds"])
        assert row["validation_r_loss"] >= 0
    targets = pd.read_csv(out / "targeting_comparison.csv")
    assert targets.groupby(["dataset", "outcome", "fraction"]).n.nunique().eq(1).all()
    rank = pd.read_csv(out / "uplift_ranking_metrics.csv")
    assert rank.groupby(["dataset", "outcome"]).endpoint_gain.nunique().eq(1).all()
    assert targets.loc[targets["dataset"].eq("local_fixture"), "mean_true_cate"].isna().all()
    manifest = json.loads((out / "causal_model_manifest.json").read_text(encoding="utf-8"))["models"]
    assert all(row["reload_verified"] and (out / row["model_path"]).is_file() for row in manifest)
