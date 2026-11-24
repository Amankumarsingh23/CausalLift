import joblib
import numpy as np
import pandas as pd
import pytest

from causallift.confounding_demo import TRUE_EFFECT, simulate_confounding
from causallift.double_ml import (
    AdvancedDML,
    orthogonal_score,
    partially_linear_dml,
    structural_effect,
)
from causallift.simulation import simulate
from causallift.uplift import RandomizedEvaluation


def test_structural_coefficient_score_derivative_and_standard_error():
    y, t = [1, 3, 0, 0], [-0.5, 0.5, -0.5, 0.5]
    estimate = structural_effect(y, t)
    assert estimate.estimate == 1
    assert estimate.denominator == 1
    assert estimate.derivative == -0.25
    assert estimate.score_mean == 0
    assert estimate.standard_error == pytest.approx(1.5)
    np.testing.assert_allclose(orthogonal_score(y, t, estimate.estimate), [-0.75, 1.25, -0.25, -0.25])
    assert estimate.ci_low < estimate.estimate < estimate.ci_high
    with pytest.raises(ValueError, match="Insufficient residual"):
        structural_effect(y, [1e-7] * 4)


def test_known_propensity_and_fold_coverage():
    rows, truth, spec = simulate("D", 600, 9)
    x = spec.features(rows.merge(truth, on="unit_id").assign(split="train", exposure=0))
    estimate, detail = partially_linear_dml(x, rows.treatment, rows.outcome, propensity=0.15)
    assert set(x).isdisjoint({"unit_id", "split", "treatment", "outcome", "tau_true", "exposure"})
    assert detail["propensity_crossfit"] is None
    assert detail["propensity_min"] == detail["propensity_max"] == 0.15
    assert sum(f["held_out"]["n"] for f in detail["outcome_folds"]) == len(rows)
    assert all(f["held_out"]["n"] + f["training"]["n"] == len(rows) for f in detail["outcome_folds"])
    assert np.isfinite(estimate.estimate)
    assert estimate.score_mean == pytest.approx(0, abs=1e-12)


def test_continuous_dml_crossfits_estimated_propensity():
    x, t, y, overlap = simulate_confounding(14, 850)
    estimate, detail = partially_linear_dml(x, t, y, binary_outcome=False, estimators=24)
    assert (0.10 <= overlap).all() and (overlap <= 0.90).all()
    assert 0 < detail["propensity_min"] < detail["propensity_max"] < 1
    assert len(detail["outcome_folds"]) == len(detail["propensity_crossfit"]["folds"]) == 3
    assert detail["outcome_metrics"]["rmse"] > 0
    assert abs(estimate.estimate - TRUE_EFFECT) < 1.0
    assert estimate.standard_error > 0


def test_nonfinite_inference_se_does_not_create_a_zero_width_interval():
    class Inference:
        stderr = np.array([0.1, np.nan])

        def conf_int(self, alpha):
            return [-0.1, 0.5], [0.3, 0.5]

    class LibraryModel:
        def effect_inference(self, X):
            return Inference()

    model = AdvancedDML("causal_forest", 0.5)
    model.feature_names_ = ("x",)
    model.model_ = LibraryModel()
    lo, hi = model.effect_interval(pd.DataFrame({"x": [1, 2]}))
    assert lo[0] == -0.1 and hi[0] == 0.3
    assert np.isnan(lo[1]) and np.isnan(hi[1])


@pytest.mark.parametrize("kind", ("linear_dml", "causal_forest"))
def test_advanced_effect_sign_library_integration_intervals_and_reload(kind, tmp_path):
    pytest.importorskip("econml")
    from econml.dml import CausalForestDML, LinearDML

    rows, _, spec = simulate("A", 1800, 17)
    train, val = rows.iloc[:1400], rows.iloc[1400:]
    x, xv = spec.features(train), spec.features(val)
    model = AdvancedDML(kind, 0.5, trees=24, min_leaf=12).fit(x, train.treatment, train.outcome)
    assert isinstance(model.model_, CausalForestDML if kind == "causal_forest" else LinearDML)
    effect = model.predict_effect(xv)
    assert effect.shape == (len(val),)
    assert effect.mean() > 0.03
    lo, hi = model.effect_interval(xv)
    assert np.all(lo <= effect) and np.all(effect <= hi)
    al, ah = model.average_interval(xv)
    assert al <= effect.mean() <= ah
    curve = RandomizedEvaluation(val.treatment, val.outcome, 0.5).curve(effect)
    assert len(curve.gain) == len(val) + 1
    path = tmp_path / "advanced.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(joblib.load(path).predict_effect(xv), effect, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="original feature order"):
        model.predict_effect(xv[xv.columns[::-1]])
