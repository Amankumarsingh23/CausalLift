from functools import partial

import numpy as np
import pandas as pd
import pytest

from causallift.crossfit import crossfit_outcome
from causallift.meta_learners import fit_binary_outcome, outcome_configuration


class HeldOutWitness:
    def __init__(self, frame):
        self.training_rows = set(frame.index)
        self.training_groups = set(frame["x"] // 4)

    def predict_proba(self, frame):
        assert self.training_rows.isdisjoint(frame.index)
        assert self.training_groups.isdisjoint(set(frame["x"] // 4))
        p = frame["x"].to_numpy() / 100
        return np.column_stack([1 - p, p])


def test_crossfit_predicts_each_row_once_with_group_isolation():
    frame = pd.DataFrame({"x": np.arange(48)})
    treatment = np.tile([0, 0, 1, 1], 12)
    outcome = np.tile([0, 1, 0, 1], 12)
    groups = frame["x"] // 4

    def fit(x, y):
        assert len(x) == len(y)
        return HeldOutWitness(x)

    result = crossfit_outcome(frame, treatment, outcome, fit, groups=groups, seed=17)
    again = crossfit_outcome(frame, treatment, outcome, fit, groups=groups, seed=17)
    np.testing.assert_array_equal(result.membership, again.membership)
    np.testing.assert_allclose(result.predictions, frame["x"] / 100)
    assert result.strategy == "group_joint"
    assert len(result.folds) == 3
    assert sum(fold["held_out"]["n"] for fold in result.folds) == len(frame)
    assert all(fold["held_out"]["control_events"] and fold["held_out"]["treated_events"] for fold in result.folds)


def test_preprocessing_fits_only_fold_training_and_rare_fallback():
    frame = pd.DataFrame({"x": np.arange(60, dtype=float)**2})
    t = np.tile([0, 1], 30)
    y = np.ones(60, dtype=int)
    y[-1] = 0
    fit = partial(fit_binary_outcome, configuration=outcome_configuration("logistic", 37, 10),
                  scaled_columns=("x",))
    result = crossfit_outcome(frame, t, y, fit, seed=17)
    assert result.strategy == "treatment"
    assert "Rare" in result.warnings[0]
    assert sorted(np.unique(result.membership)) == [0, 1, 2]
    for i, model in enumerate(result.models):
        in_training = result.membership != i
        if isinstance(model, float):
            assert model == 1
            assert result.folds[i]["constant_fallback"] == 1
        else:
            scale = model.named_steps["columntransformer"].named_transformers_["features"]
            assert scale.mean_[0] == pytest.approx(frame.loc[in_training, "x"].mean())
    assert any(isinstance(model, float) for model in result.models)


def test_unstratified_group_fallback_preserves_groups():
    frame = pd.DataFrame({"x": np.arange(24)})
    groups = frame["x"] // 4
    t = np.zeros(len(frame), dtype=int)
    t[-1] = 1
    y = t.copy()
    result = crossfit_outcome(frame, t, y, lambda x, y: float(y.mean()), groups=groups)
    assert result.strategy == "group_unstratified"
    assert pd.Series(result.membership).groupby(groups).nunique().eq(1).all()
    with pytest.raises(ValueError, match="Too few distinct groups"):
        crossfit_outcome(frame, t, y, lambda x, y: 0.0, groups=np.zeros(len(y)))
