import joblib
import numpy as np
import pandas as pd
import pytest

from causallift.meta_learners import ASSIGNMENT, MetaLearner
from causallift.simulation import simulate


@pytest.mark.parametrize("base", ["logistic", "lightgbm"])
def test_s_counterfactual_toggle_and_sign(base):
    x = pd.DataFrame({"x": np.tile(np.repeat(np.arange(10), 20), 2)})
    t = np.repeat([0, 1], 200)
    y = np.r_[np.tile([1, 0, 0, 0, 0], 40), np.tile([0, 1, 1, 1, 1], 40)]
    model = MetaLearner("s", base, estimators=30).fit(x, t, y)
    result = model.predict_potential_outcomes(x)
    for assignment, values in enumerate((result.mu0, result.mu1)):
        assert np.all((0 <= values) & (values <= 1))
        np.testing.assert_allclose(values, model.models_[0].predict_proba(x.assign(**{ASSIGNMENT: assignment}))[:, 1])
    assert (result.effect > 0).all()


@pytest.mark.parametrize("base", ["logistic", "lightgbm"])
def test_t_arm_specific_training_and_single_class_fallback(base):
    x = pd.DataFrame({"x": np.arange(100)})
    t = np.repeat([0, 1], 50)
    y = np.r_[np.zeros(50), np.tile([0, 1], 25)]
    model = MetaLearner("t", base, estimators=15).fit(x, t, y)
    result = model.predict_potential_outcomes(x)
    assert model.fallbacks_ == {"control": 0.0}
    assert model.arm_stats_ == {"control": {"n": 50, "positives": 0}, "treated": {"n": 50, "positives": 25}}
    assert np.all(result.mu0 == 0)
    np.testing.assert_allclose(result.mu1, model.models_[1].predict_proba(x)[:, 1])
    np.testing.assert_allclose(result.effect, model.predict_effect(x))
    if base == "logistic":
        scale = model.models_[1].named_steps["columntransformer"].named_transformers_["features"]
        assert scale.mean_[0] == x.loc[t == 1, "x"].mean()


def test_feature_contract_and_serialization(tmp_path):
    observed, truth, spec = simulate("E", 600)
    joined = observed.merge(truth, on="unit_id").assign(exposure=1, split="train")
    x = spec.features(joined)
    assert set(x).isdisjoint({"tau_true", "treatment", "outcome", "unit_id", "exposure", "split"})
    model = MetaLearner("t", "logistic").fit(x, joined["treatment"], joined["outcome"])
    path = tmp_path / "meta.joblib"
    joblib.dump(model, path)
    np.testing.assert_array_equal(model.predict_effect(x), joblib.load(path).predict_effect(x))
    with pytest.raises(ValueError, match="original order"):
        model.predict_effect(x[x.columns[::-1]])
    with pytest.raises(ValueError, match="Both treatment arms"):
        model.fit(x, np.zeros(len(x)), joined["outcome"])
    with pytest.raises(ValueError, match="Binary outcomes"):
        model.fit(x, joined["treatment"], np.full(len(x), 2))
    with pytest.raises(ValueError, match="equal lengths"):
        model.fit(x.iloc[:-1], joined["treatment"], joined["outcome"])
