import json
from dataclasses import replace

import joblib
import numpy as np
import pandas as pd
import pytest

from causallift.causal_dags import CausalDAG
from causallift.propensity import fit_assignments, overlap, propensity_prediction
from causallift.session08 import bias_demonstrations, graph_examples, run


def test_backdoor_and_graph_validation(tmp_path):
    graphs = graph_examples()
    confounder = graphs["confounder"]
    assert not confounder.backdoor(())["identified"]
    assert confounder.backdoor()["identified"]
    assert graphs["mediator"].backdoor()["identified"]
    assert graphs["mediator"].backdoor(("M",))["invalid"] == ["M"]
    assert graphs["collider"].backdoor()["backdoor_d_separated"]
    assert not graphs["collider"].backdoor(("C",))["identified"]
    assert not graphs["hidden"].backdoor()["identified"]
    assert graphs["hidden"].backdoor(("U",))["invalid"] == ["U"]
    with pytest.raises(ValueError, match="acyclic"):
        replace(confounder, edges=(*confounder.edges, ("Y", "X")))
    with pytest.raises(ValueError, match="observed"):
        replace(graphs["hidden"], adjustment=("U",))
    path = tmp_path / "dag.json"
    confounder.save(path)
    assert CausalDAG.load(path) == confounder


def test_simulated_bias_and_overlap():
    values = {(r["case"], r["analysis"]): r["coefficient"] for r in bias_demonstrations(42, 20000)}
    assert values["confounder", "adjusted"] == pytest.approx(1, abs=.07)
    assert values["mediator", "unadjusted"] == pytest.approx(4, abs=.10)
    assert values["mediator", "adjusted"] == pytest.approx(1, abs=.10)
    assert values["collider", "adjusted"] < .15
    row = overlap([0, 0, 1, 1], [0, .6, .4, 1])
    assert (row["support_low"], row["support_high"], row["outside_support_fraction"]) == (.4, .6, .5)
    assert row["below_threshold"] == row["above_threshold"] == .25
    assert "Constant" in overlap([0, 1], [.5, .5])["warnings"]
    assert overlap([0, 1], [0, 1])["outside_support_fraction"] == 1


def test_assignment_target_and_group_fold_alignment():
    x = pd.DataFrame({"x": np.linspace(-1, 1, 120)})
    t = np.tile([0, 1], 60)
    groups = np.arange(120)//4
    for method in ("constant", "logistic", "lightgbm"):
        model, cross, _ = fit_assignments(x, t, method, groups=groups)
        _, repeat, _ = fit_assignments(x, t, method, groups=groups)
        assert pd.Series(cross.membership).groupby(groups).nunique().eq(1).all()
        np.testing.assert_array_equal(cross.membership, repeat.membership)
        assert np.isfinite(cross.predictions).all()
        p = propensity_prediction(model, x)
        assert np.all((0 <= p) & (p <= 1))
    np.testing.assert_array_equal( propensity_prediction(fit_assignments(x, t, "constant")[0], x), .5)


def test_offline_reports_and_roundtrip(tmp_path):
    summary = run(tmp_path, fixture=True)
    out = tmp_path / "artifacts/session_08_fixture"
    assert summary["models"] == 9
    manifest = json.loads((out / "propensity_predictions_manifest.json").read_text())["models"]
    graphs = json.loads((out / "identification_results.json").read_text())
    assert graphs["collider"]["path_demo_without_direct_effect_edge"] == {
        "unconditioned_d_separated": True, "conditioned_d_separated": False,
    }
    for entry in manifest:
        assert entry["target"] == "treatment"
        assert not {"treatment", "outcome", "latent_confounder", "split", "propensity_true"} & set(entry["feature_allowlist"])
        table = pd.read_parquet(out / entry["prediction_path"])
        assert table.groupby("partition").position.apply(lambda p: np.array_equal(p, np.arange(len(p)))).all()
        assert table.loc[table.partition.eq("train"), "fold"].nunique() == 3
        assert table.propensity.between(0, 1).all()
        assert joblib.load(out / entry["model_path"]) is not None
    data = pd.read_csv(out / "propensity_model_results.csv")
    assert data.loc[data.scenario.eq("F"), "true_propensity_rmse"].isna().all()
    assert data.loc[data.scenario.eq("C"), "true_propensity_rmse"].notna().all()
    constant = data.loc[data.model.eq("constant")]
    np.testing.assert_allclose(constant.brier_score, constant.prevalence * (1-constant.mean_probability)**2
                               + (1-constant.prevalence) * constant.mean_probability**2)
