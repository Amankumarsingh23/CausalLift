import numpy as np
import pandas as pd
import pytest

from causallift.simulation import FEATURES, SCENARIOS, TRUTH, simulate


def test_determinism():
    first, first_truth, first_spec = simulate("F", 100, 5)
    second, second_truth, second_spec = simulate("F", 100, 5)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_truth, second_truth)
    assert first_spec == second_spec
    assert not simulate("F", 100, 6)[0].equals(first)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_simulation_invariants(scenario):
    observed, truth, spec = simulate(scenario, 1200)
    assert observed.shape == (1200, 9)
    assert list(observed) == [*FEATURES, "unit_id", "treatment", "outcome"]
    assert list(truth) == ["unit_id", *spec.ground_truth_columns]
    assert set(TRUTH).issubset(truth)
    assert observed["unit_id"].is_unique
    pd.testing.assert_series_equal(observed["unit_id"], truth["unit_id"])
    for column in ("treatment", "outcome"):
        assert set(observed[column]) == {0, 1}
    for column in ("y0_true", "y1_true"):
        assert set(truth[column]) == {0, 1}
    for column in ("mu0_true", "mu1_true", "propensity_true"):
        assert truth[column].between(0, 1).all()
    np.testing.assert_allclose(truth["tau_true"], truth["mu1_true"] - truth["mu0_true"])
    np.testing.assert_array_equal(observed["outcome"], np.where(
        observed["treatment"], truth["y1_true"], truth["y0_true"],
    ))
    assert list(spec.features(observed.merge(truth))) == list(FEATURES)


def test_constant_effect():
    _, truth, spec = simulate("A")
    np.testing.assert_allclose(truth["tau_true"], 0.1, atol=1e-15)
    assert truth["propensity_true"].eq(0.5).all()
    assert spec.assignment_probability == 0.5


@pytest.mark.parametrize("scenario", ("B", "E"))
def test_effect_signs(scenario):
    _, truth, _ = simulate(scenario)
    assert truth["tau_true"].min() < -0.1
    assert truth["tau_true"].max() > 0.1


def test_imbalanced_assignment():
    frame, truth, spec = simulate("D")
    assert frame["treatment"].mean() == pytest.approx(0.15, abs=0.015)
    assert truth["propensity_true"].eq(0.15).all()
    assert spec.assignment_probability == 0.15


def test_observed_confounding():
    observed, truth, spec = simulate("C")
    assert observed["engagement"].corr(truth["propensity_true"]) > 0.7
    assert observed["engagement"].corr(truth["mu0_true"]) > 0.6
    assert spec.assignment_probability is None


def test_hidden_confounding_and_overlap():
    observed, truth, spec = simulate("F")
    assert "latent_confounder" not in observed
    assert "latent_confounder" not in spec.features(observed.merge(truth))
    assert truth["latent_confounder"].corr(truth["mu0_true"]) > 0.6
    assert truth["latent_confounder"].corr(truth["propensity_true"]) > 0.5
    assert truth["propensity_true"].between(0.05, 0.95).mean() < 0.6
    assert abs((truth["tau_true"] - truth["tau_observed_true"]).mean()) < 0.003
