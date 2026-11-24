import json

import pandas as pd

from causallift.session01 import persist
from causallift.session02 import run
from causallift.simulation import SCENARIOS, simulate
from tests.test_data import FIXTURE_PATH


def test_fixture_report_without_real_data(tmp_path):
    results = run(tmp_path, mode="fixture", fixture=FIXTURE_PATH, bootstrap_resamples=30)
    output = tmp_path / "artifacts/session_02_fixture"
    assert all(row["interpretation"] == "Fixture demonstration" for row in results["criteo_comparisons"])
    assert len(pd.read_csv(output / "fixture_experiment_results.csv")) == 2
    continuous = json.loads((output / "continuous_experiment_results.json").read_text(encoding="utf-8"))
    assert continuous["test_method"] == "welch_t"
    assert continuous["bootstrap_interval"] is not None
    assert (output / "treatment_effect_intervals.png").read_bytes().startswith(b"\x89PNG")


def test_synthetic_reports_preserve_identification_labels(tmp_path):
    synthetic = tmp_path / "data/synthetic"
    synthetic.mkdir(parents=True)
    for scenario in SCENARIOS:
        observed, truth, spec = simulate(scenario, 500)
        path = synthetic / f"{spec.dataset_id}.parquet"
        persist(observed, spec, path, 42)
        truth.to_parquet(path.with_stem(path.stem + "_truth"), index=False)
    result = run(tmp_path, mode="synthetic", bootstrap_resamples=0)
    output = tmp_path / "artifacts/session_02_synthetic"
    report = pd.read_csv(output / "synthetic_experiment_results.csv")
    assert len(report) == 6
    assert report["total_n"].eq(500).all()
    assert set(report.loc[~report["identifies_ate"], "scenario"]) == {"C", "F"}
    assert set(report.loc[report["identifies_ate"], "scenario"]) == {"A", "B", "D", "E"}
    for row in result["synthetic_comparisons"]:
        assert row["estimation_error"] == row["absolute_lift"] - row["true_ate"]
        if not row["identifies_ate"]:
            assert row["interpretation"].startswith("Observational association")
    subgroups = pd.read_csv(output / "subgroup_analysis.csv")
    assert subgroups["partition"].eq("validation").all()
    assert subgroups["exploratory"].all()
    assert (output / "experiment_summary.json").is_file()
