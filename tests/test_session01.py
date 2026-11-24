import json

import pandas as pd

from causallift.session01 import run
from tests.test_data import FIXTURE_PATH


def test_end_to_end_local_fixture(tmp_path):
    run(tmp_path, dataset=FIXTURE_PATH, fixture=True, rows=350)
    artifacts = tmp_path / "artifacts/session_01"
    summary = json.loads((artifacts / "dataset_summary.json").read_text(encoding="utf-8"))
    assert len(summary["datasets"]) == 7
    fixture = summary["datasets"]["local_fixture"]
    assert fixture["source_rows"] == 60
    assert fixture["spec"]["source_type"] == "fixture"
    assert fixture["publisher_archive_verified"] is False
    report = pd.read_csv(artifacts / "synthetic_scenario_summary.csv")
    assert set(report["scenario"]) == set("ABCDEF")
    assert report["rows"].eq(350).all()
    splits = pd.read_csv(artifacts / "split_summary.csv")
    assert len(splits) == 42
    for name in ("synthetic_effect_distribution.png", "propensity_overlap.png"):
        assert (artifacts / name).read_bytes().startswith(b"\x89PNG")
    observed = pd.read_parquet(tmp_path / "data/synthetic/synthetic_F.parquet")
    truth = pd.read_parquet(tmp_path / "data/synthetic/synthetic_F_truth.parquet")
    assert set(observed).intersection(truth) == {"unit_id"}
    assert (artifacts / "data_quality_report.json").is_file()
