import json

import numpy as np
import pandas as pd
import pytest

from causallift.data.splits import split_data
from causallift.observational import (
    aipw,
    effective_n,
    ht,
    match,
    overlap_profiles,
    standardize,
    weighted_contrast,
    weighted_smd,
    weights,
)
from causallift.prediction import training_fingerprint
from causallift.propensity import fit_assignments, propensity_prediction
from causallift.session09 import _outcome_oof, load_saved_propensity, run
from causallift.simulation import simulate


def test_formula_with_continuous_outcomes():
    t, y, e = (
        np.array([False, True, False, True]),
        np.array([2.0, 7.0, 4.0, 9.0]),
        [0.25, 0.5, 0.5, 0.75],
    )
    w = weights(t, e)
    np.testing.assert_allclose(w, [4 / 3, 2, 2, 4 / 3])
    assert ht(t, y, e) == pytest.approx((14 + 12 - 8 - 8 / 3) / 4)
    assert weighted_contrast(t, y, w) == pytest.approx(4.6)
    att = weights(t, e, "att")
    np.testing.assert_allclose(att, [1 / 3, 1, 1, 1])
    assert weighted_contrast(t, y, att) == pytest.approx(4.5)
    assert effective_n([1, 1, 2]) == pytest.approx(16 / 6)
    assert weighted_smd(t, y, w, denominator=2) == pytest.approx(2.3)
    assert standardize([1, 2], [3, 8], [True, False]) == 2
    a, b = np.full(4, 3), np.full(4, 7)
    score = np.array([4 + 4 / 3, 4, 2, 4 + 8 / 3])
    result = aipw(t, y, e, a, b)
    assert result["estimate"] == pytest.approx(score.mean())
    assert result["score_variance"] == pytest.approx(score.var(ddof=1))
    assert result["se"] == pytest.approx(score.std(ddof=1) / 2)


def test_matching_and_overlap_safeguards():
    t = [0, 0, 1, 1, 1]
    e = [0.4, 0.4, 0.4, 0.4, 0.95]
    tr, co, dist, _ = match(t, e)
    np.testing.assert_array_equal(tr, [2, 3])
    np.testing.assert_array_equal(co, [0, 0])
    np.testing.assert_array_equal(dist, [0, 0])
    profiles = overlap_profiles([0.01, 0.5, 0.99])
    assert profiles["clip_05_95"][0].all()
    np.testing.assert_allclose(profiles["clip_05_95"][1], [0.05, 0.5, 0.95])
    np.testing.assert_array_equal(profiles["restrict_05_95"][0], [False, True, False])
    with pytest.raises(ValueError, match="interior"):
        weights([0, 1], [0, 1])
    with pytest.raises(ValueError, match="positive"):
        weighted_contrast(np.array([0, 1]), [0, 1], [0, 1])


def test_propensity_reuse_verifies_alignment_and_outcome_folds(tmp_path):
    frame, _, spec = simulate("C", 450, 42)
    frame["split"] = split_data(frame, spec, 42)
    train, val = [frame.loc[frame.split.eq(part)] for part in ("train", "validation")]
    model, cross, config = fit_assignments(
        spec.features(train), train.treatment, "logistic", groups=train.unit_id
    )
    saved = pd.concat(
        [
            pd.DataFrame(
                {
                    "partition": name,
                    "position": np.arange(len(part)),
                    "unit_id": part.unit_id.to_numpy(),
                    "fold": fold,
                    "propensity": p,
                }
            )
            for name, part, fold, p in (
                ("train", train, cross.membership, cross.predictions),
                (
                    "validation",
                    val,
                    -1,
                    propensity_prediction(model, spec.features(val)),
                ),
            )
        ]
    )
    path = tmp_path / "scores.parquet"
    saved.to_parquet(path, index=False)
    record = {
        "scenario": "C",
        "model": "logistic",
        "target": "treatment",
        "feature_allowlist": list(spec.feature_columns),
        "configuration": config,
        "seed": 42,
        "row_identity": "unit_id",
        "source": "data/synthetic/synthetic_C.parquet",
        "prediction_path": path.name,
        "train_fingerprint": training_fingerprint(train, spec, "treatment"),
        "validation_fingerprint": training_fingerprint(val, spec, "treatment"),
    }
    (tmp_path / "propensity_predictions_manifest.json").write_text(
        json.dumps({"models": [record]}), encoding="utf-8"
    )
    _, _, fold = load_saved_propensity(tmp_path, "C", spec, train, val, 42)
    np.testing.assert_array_equal(_outcome_oof(train, spec, 42)[2], fold)
    with pytest.raises(ValueError, match="alignment"):
        load_saved_propensity(tmp_path, "C", spec, train, val.iloc[::-1], 42)


def test_offline_outputs_preserve_estimands(tmp_path):
    summary = run(tmp_path, fixture=True)
    target = tmp_path / "artifacts/session_09_fixture"
    att = pd.read_csv(target / "observational_att_results.csv")
    diagnostic = pd.read_csv(target / "matching_diagnostics.csv")
    matched = att.loc[att.estimand.eq("matched ATT")]
    assert matched.n.tolist() == diagnostic.matched_treated.tolist()
    assert matched.truth.notna().all()
    ate = pd.read_csv(target / "observational_ate_results.csv")
    assert set(ate.population) == {"validation", "training OOF"}
    assert summary["nsw"]["status"] == "offline fixture"
    assert (target / "covariate_balance_love_plot.png").is_file()
