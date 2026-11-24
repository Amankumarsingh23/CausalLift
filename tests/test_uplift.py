import numpy as np
import pytest

from causallift.targeting import rank_scores
from causallift.uplift import RandomizedEvaluation, effect_accuracy


def test_effect_accuracy_against_hand_fixture():
    answer = effect_accuracy([0.3, -0.1], [0.1, 0.0])
    assert answer["root_pehe"] == pytest.approx(np.sqrt(0.025))
    assert answer["cate_mae"] == pytest.approx(0.15)
    assert answer["signed_ate_error"] == pytest.approx(0.05)
    assert answer["absolute_ate_error"] == pytest.approx(0.05)
    assert answer["spearman"] == pytest.approx(1)
    assert effect_accuracy([0.2, 0.1], [0.1, 0.1 + 1e-17])["spearman"] is None
    assert effect_accuracy([0.1, 0.0], [0.3, 0.1])["signed_ate_error"] == pytest.approx(-0.15)


def test_gain_uplift_qini_and_area_match_hand_calculation():
    evaluation = RandomizedEvaluation([1, 0, 1, 0], [1, 0, 0, 1], 0.5)
    ranks = np.arange(4)
    curve = evaluation.curve_from_order(ranks, truth=[0.1, 0.2, -0.1, -0.2])
    np.testing.assert_array_equal(curve.fraction, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_array_equal(curve.gain, [0, 0.5, 0.5, 0.5, 0])
    np.testing.assert_array_equal(curve.qini, [0, np.nan, 0.25, 0.25, 0])
    np.testing.assert_allclose(curve.true_gain, [0, 0.025, 0.075, 0.05, 0], atol=1e-15)
    assert evaluation.summarize_selection(ranks[:2])["uplift"] == 1
    assert curve.metrics()["auuc"] == pytest.approx(0.375)
    assert curve.metrics()["qini_coefficient"] == pytest.approx(0.15625)
    assert evaluation.summarize_selection(ranks[:1])["uplift"] is None
    assert evaluation.summarize_selection(ranks[:0])["n"] == 0


def test_unequal_propensities_random_reference_and_invariant_endpoint():
    evaluation = RandomizedEvaluation([1, 0, 1, 0], [1, 1, 0, 0], [0.25, 0.5, 0.25, 0.5])
    curve = evaluation.curve([4, 3, 2, 1])
    np.testing.assert_array_equal(curve.gain, [0, 1, 0.5, 0.5, 0.5])
    np.testing.assert_array_equal(curve.random_gain, [0, 0.125, 0.25, 0.375, 0.5])
    assert curve.metrics()["gain_area_above_random"] == pytest.approx(curve.metrics()["auuc"] - 0.25)
    reverse = evaluation.curve([1, 2, 3, 4])
    assert reverse.gain[-1] == curve.gain[-1]
    assert reverse.qini[-1] == curve.qini[-1]
    with pytest.raises(ValueError, match="strictly"):
        RandomizedEvaluation([0, 1], [0, 1], 1)
    with pytest.raises(ValueError, match="finite"):
        evaluation.curve([0, 1, 2, np.nan])


def test_ties_and_calibration_counts():
    t = np.tile([0, 1], 10)
    a = RandomizedEvaluation(t, np.ones(20), 0.5)
    b = RandomizedEvaluation(t, np.zeros(20), 0.5)
    scores = np.repeat([0.1, 0.2], 10)
    ranking = rank_scores(scores, seed=9)
    np.testing.assert_array_equal(a.curve(scores, seed=9).treated_n, b.curve(scores, seed=9).treated_n)
    np.testing.assert_array_equal(a.curve(scores, seed=9).treated_n, a.curve_from_order(ranking).treated_n)
    groups = a.calibration(scores)
    assert [(r["n"], r["treated_n"], r["control_events"]) for r in groups] == [(10, 5, 5), (10, 5, 5)]
    assert len(b.calibration(np.zeros(20))) == 1
    assert b.calibration(scores)[0]["uplift"] is None
    assert b.summarize_selection(ranking)["uplift"] == 0
    assert b.summarize_selection(ranking)["effect_se"] is None
