import json

import pandas as pd

from causallift.session05 import run
from tests.test_data import FIXTURE_PATH


def test_session05_offline_fixture_includes_models_equal_capacities_and_truth_isolation(tmp_path):
    fixture = tmp_path / "tests/fixtures/criteo_schema.csv"
    fixture.parent.mkdir(parents=True)
    fixture.write_bytes(FIXTURE_PATH.read_bytes())
    summary = run(tmp_path, fixture=True)
    assert summary["models"] == 12
    assert summary["evaluated_partitions"] == ["validation"]
    output = tmp_path / "artifacts/session_05_fixture"
    target = pd.read_csv(output / "targeting_comparison.csv")
    assert target.groupby(["dataset", "outcome", "fraction"])["n"].nunique().eq(1).all()
    real_fixture = target.loc[target["dataset"].eq("local_fixture")]
    assert real_fixture["expected_incremental_conversions"].isna().all()
    assert not real_fixture["policy"].eq("oracle").any()
    synthetic = target.loc[target["dataset"].eq("synthetic_fixture_E")]
    assert synthetic["policy"].eq("oracle").any()
    manifest = json.loads((output / "causal_model_manifest.json").read_text(encoding="utf-8"))["models"]
    for model in manifest:
        assert model["reload_verified"]
        assert set(model["feature_allowlist"]).isdisjoint({"unit_id", "exposure", "outcome", "treatment", "tau_true"})
        assert (output / model["model_path"]).is_file()
    for name in ("uplift_curves.png", "qini_curves.png", "uplift_calibration.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG")
