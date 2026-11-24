import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from causallift.prediction import (
    calibration_bins,
    evaluate,
    fit_model,
    training_fingerprint,
)
from causallift.simulation import simulate


def data():
    frame, truth, spec = simulate("E", 600, 31)
    return frame.iloc[:400].assign(split="train"), frame.iloc[400:].assign(split="validation"), truth, spec


def test_training_prevalence_constant_scores():
    train, validation, _, spec = data()
    model = fit_model("prevalence", spec, train, validation, "outcome")
    np.testing.assert_array_equal(model.predict(validation), np.full(len(validation), train.outcome.mean()))
    all_zero = fit_model("prevalence", spec, train.assign(outcome=0), validation, "outcome")
    np.testing.assert_array_equal(all_zero.predict(validation), np.zeros(len(validation)))


@pytest.mark.parametrize("model_name", ("logistic", "lightgbm"))
def test_probabilities_reproducible_and_truth_excluded(model_name):
    train, validation, truth, spec = data()
    extras = validation.merge(truth, on="unit_id").assign(latent_confounder=999)
    models = [fit_model(model_name, spec, sample, val, "outcome", estimators=25)
              for sample, val in ((train, validation), (train.merge(truth, on="unit_id"), extras))]
    p = models[0].predict(validation)
    assert p.shape == (len(validation),)
    assert np.all((0 <= p) & (p <= 1))
    np.testing.assert_allclose(p, models[1].predict(extras.assign(treatment=0, outcome=0)), atol=1e-14)
    assert tuple(models[0].estimator.feature_names_in_) == spec.feature_columns


def test_scaler_fits_only_train_and_partitions_enforced():
    train, validation, _, spec = data()
    shifted = validation.assign(engagement=validation.engagement + 1000)
    model = fit_model("logistic", spec, train, shifted, "outcome")
    np.testing.assert_allclose(model.estimator.named_steps["standardscaler"].mean_, spec.features(train).mean())
    for bad in ("test", "train"):
        with pytest.raises(ValueError, match="validation"):
            fit_model("logistic", spec, train, validation.assign(split=bad), "outcome")
    with pytest.raises(ValueError, match="configured"):
        fit_model("logistic", spec, train, validation, "treatment")
    assert training_fingerprint(train, spec, "outcome") == training_fingerprint(train.assign(exposure=1), spec, "outcome")
    assert training_fingerprint(train, spec, "outcome") != training_fingerprint(train.assign(outcome=0), spec, "outcome")


def test_metrics_match_sklearn_and_single_class_has_no_ranking_metrics():
    y = [0, 0, 1, 0, 1]
    p = [0.05, 0.2, 0.15, 0.4, 0.9]
    metrics = evaluate(y, p)
    for name, function in (("roc_auc", roc_auc_score), ("average_precision", average_precision_score),
                           ("log_loss", log_loss), ("brier_score", brier_score_loss)):
        assert metrics[name] == pytest.approx(function(y, p))
    single = evaluate([0, 0], [0.1, 0.2])
    assert single["roc_auc"] is single["average_precision"] is None
    assert single["log_loss"] > 0 and single["warning"]
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.1, np.nan])


def test_calibration_bins_counts_and_constant_ties():
    y, p = [0, 1, 0, 1], [0.01, 0.01, 0.02, 0.04]
    bins = calibration_bins(y, p, bins=2)
    assert sum(row["count"] for row in bins) == 4
    assert sum(row["positives"] for row in bins) == 2
    assert bins[0]["mean_probability"] == 0.01
    assert bins[0]["observed_rate"] == 0.5
    constant = calibration_bins(y, [0.02] * 4)
    assert len(constant) == 1
    assert constant[0]["count"] == 4
