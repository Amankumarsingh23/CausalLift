import joblib
import numpy as np
import pandas as pd
import pytest

from causallift.data.criteo import CRITEO
from causallift.residual_learners import (
    ConstantPropensity,
    ResidualLearner,
    residuals,
    training_propensity,
    weighted_r_targets,
    x_pseudo_effects,
)
from causallift.simulation import simulate


def test_x_pseudo_effects_and_actual_weighted_prediction():
    t, y = [0, 0, 1, 1], [0, 1, 0, 1]
    outcome0, outcome1 = [0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]
    np.testing.assert_allclose(x_pseudo_effects(t, y, outcome0, outcome1), [0.6, -0.3, -0.3, 0.6])
    frame = pd.DataFrame({"x": np.r_[np.tile(range(4), 5), np.tile(range(4), 5)]})
    assignment = np.repeat([0, 1], 20)
    outcome = np.r_[np.zeros(20), np.ones(20)]
    model = ResidualLearner("x", "linear", ConstantPropensity(0.8, "fixture")).fit(frame, assignment, outcome)
    np.testing.assert_allclose(model.predict_effect(frame), np.ones(40))
    assert model.diagnostics_["outcome_fallbacks"] == {"control": 0.0, "treated": 1.0}
    observed, _, spec = simulate("D", 600)
    x = spec.features(observed)
    model = ResidualLearner("x", "linear", ConstantPropensity(0.15, "design")).fit(x, observed.treatment, observed.outcome)
    a, b = [arm.predict(x) for arm in model.effect_models_]
    assert not np.allclose(a, b)
    np.testing.assert_allclose(model.predict_effect(x), 0.15 * a + 0.85 * b)


def test_r_loss_targets_residuals_and_weight_equivalence():
    yr, tr = residuals([0, 1, 1], [1, 1, 0], [0.25, 0.5, 0.25], 0.2)
    np.testing.assert_allclose(yr, [0.75, 0.5, -0.25])
    np.testing.assert_allclose(tr, [-0.2, 0.8, 0.8])
    target, weight = weighted_r_targets(yr, tr)
    prediction = np.array([0.2, -0.4, 0.1])
    np.testing.assert_allclose(weight * (target - prediction)**2, (yr - tr * prediction)**2)
    with pytest.raises(ValueError, match="too close to zero"):
        weighted_r_targets([0, 1], [1e-12, -1])


def test_propensity_source_uses_design_or_train_and_rejects_invalid():
    frame, _, spec = simulate("D", 80)
    train = frame.assign(split="train", treatment=np.tile([0, 1], 40))
    assert training_propensity(spec, train).value == 0.15
    assert training_propensity(CRITEO, train).value == 0.5
    with pytest.raises(ValueError, match="training rows"):
        training_propensity(spec, train.assign(split="validation"))
    with pytest.raises(ValueError, match="strictly"):
        ConstantPropensity(1, "fixture")
    with pytest.raises(ValueError, match="strictly"):
        residuals([0, 1], [0, 1], [0.2, 0.4], np.nan)


@pytest.mark.parametrize("learner,base", [("x", "lightgbm"), ("r", "linear"), ("r", "lightgbm")])
def test_regression_prediction_truth_exclusion_and_reload(tmp_path, learner, base):
    frame, truth, spec = simulate("E", 700)
    train = frame.iloc[:510].merge(truth, on="unit_id").assign(split="train", exposure=1)
    val = frame.iloc[510:]
    features = spec.features(train)
    assert set(features).isdisjoint({"treatment", "outcome", "tau_true", "exposure", "unit_id", "split"})
    model = ResidualLearner(learner, base, training_propensity(spec, train), estimators=15)
    model.fit(features, train.treatment, train.outcome)
    xv = spec.features(val)
    effect = model.predict_effect(xv)
    assert effect.shape == (len(val),)
    assert np.isfinite(effect).all() and effect.std() > 0
    path = tmp_path / "learner.joblib"
    joblib.dump(model, path)
    np.testing.assert_allclose(effect, joblib.load(path).predict_effect(xv), atol=1e-14)
    if learner == "r":
        assert model.diagnostics_["covered_rows"] == len(train)
        assert model.validation_r_loss(xv, val.treatment, val.outcome) >= 0
