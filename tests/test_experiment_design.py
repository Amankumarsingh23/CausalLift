import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest, chisquare
from statsmodels.stats.multitest import multipletests

from causallift.cuped import analyze_cuped, fit_cuped
from causallift.data.criteo import CRITEO
from causallift.experiment_decisions import ExperimentSpec, Guardrail, decide_experiment
from causallift.experiment_health import (
    adjust_p_values,
    covariate_balance,
    sample_ratio_mismatch,
)
from causallift.experimentation import binary_from_counts


def test_srm_cases_and_reference():
    mismatched = sample_ratio_mismatch(650, 350, 0.5)
    reference = chisquare([650, 350], [500, 500])
    assert mismatched.alert
    assert (mismatched.chi_square, mismatched.p_value) == pytest.approx((reference.statistic, reference.pvalue))
    unequal = sample_ratio_mismatch(850, 150, 0.15)
    assert unequal.p_value == 1 and unequal.alert is False
    unknown = sample_ratio_mismatch(850, 150)
    assert unknown.status == "not_assessable"
    assert unknown.expected_treated_n is unknown.p_value is unknown.alert is None
    sparse = sample_ratio_mismatch(8, 0, 0.25)
    assert sparse.method == "exact_binomial"
    assert sparse.p_value == pytest.approx(binomtest(0, 8, 0.25).pvalue)
    with pytest.raises(ValueError, match="at least one"):
        sample_ratio_mismatch(0, 0, 0.5)


def test_smd_numeric_and_zero_variance():
    frame = pd.DataFrame({"treatment": [0, 0, 0, 1, 1], "f0": [0., 1., 2., 2., 4.],
                          "f1": [1., 1., 1., 2., 2.], "f2": [1.] * 5})
    smd, different, identical = covariate_balance(frame, CRITEO, ("f0", "f1", "f2"))
    assert smd["control_mean"] == 1 and smd["treated_mean"] == 3
    assert smd["smd"] == pytest.approx(2 / np.sqrt(1.5))
    assert smd["flag"]
    assert different["smd"] is None and different["flag"]
    assert identical["smd"] is None and identical["flag"] is None
    assert "zero variance" in identical["warnings"]
    frame.loc[0, "f0"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        covariate_balance(frame, CRITEO, ("f0",))
    with pytest.raises(ValueError, match="pre-treatment"):
        covariate_balance(frame, CRITEO, ("exposure",))


@pytest.mark.parametrize(("method", "reference", "hand"), [
    ("bonferroni", "bonferroni", (0.12, 0.03, 0.15)),
    ("holm", "holm", (0.08, 0.03, 0.08)),
    ("benjamini_hochberg", "fdr_bh", (0.05, 0.03, 0.05)),
])
def test_multiplicity_matches_statsmodels_and_hand_calculations(method, reference, hand):
    inputs = [0.04, 0.01, 0.05]
    result = adjust_p_values(inputs, method)
    rejected, corrected, *_ = multipletests(inputs, method=reference)
    assert result.raw_p_values == tuple(inputs)
    assert result.adjusted_p_values == pytest.approx(hand)
    assert result.adjusted_p_values == pytest.approx(corrected)
    assert result.reject == tuple(rejected)
    for invalid in ([], [np.nan], [-0.1], [1.01], [[0.1]]):
        with pytest.raises(ValueError):
            adjust_p_values(invalid, method)


def test_guardrail_hierarchy_and_direction():
    spec = ExperimentSpec("conversion", "higher", minimum_practical_effect=0.005,
                          guardrails=(Guardrail("refund", "lower", 0.005),))
    primary = binary_from_counts(1200, 10000, 1000, 10000)
    base = {"conversion": primary}

    def decision(refund_events, health=None):
        refund = binary_from_counts(refund_events, 10000, 200, 10000)
        return decide_experiment(spec, base | {"refund": refund}, health=health)

    assert decision(200)["decision"] == "ship"
    assert decision(350)["decision"] == "do_not_ship"
    uncertain = decision(220)
    assert uncertain["decision"] == "inconclusive"
    assert uncertain["guardrails"]["refund"]["status"] == "inconclusive"
    assert decision(350, health="fail")["decision"] == "investigate"
    assert decision(200, health="not_assessable")["decision"] == "investigate"
    assert decide_experiment(spec, base)["decision"] == "inconclusive"
    weak_primary = binary_from_counts(999, 10000, 1000, 10000)
    assert decide_experiment(spec, {"conversion": weak_primary})["decision"] == "inconclusive"

    lower = ExperimentSpec("loss", "lower")
    assert decide_experiment(lower, {"loss": binary_from_counts(600, 10000, 1000, 10000)})["decision"] == "ship"
    assert decide_experiment(lower, {"loss": primary})["decision"] == "do_not_ship"


def test_cuped_formula_constant_and_variance_reduction():
    pilot_x = np.array([-2., 0., 2., 4.])
    fit = fit_cuped(pilot_x, 10 + 3 * pilot_x)
    assert fit.theta == 3
    assert fit.pilot_correlation == pytest.approx(1)
    assert fit.adjust([0., 2.], [10., 16.]) == pytest.approx([13., 13.])
    constant = fit_cuped([2, 2, 2], [3, 5, 9])
    assert constant.theta == 0 and constant.pilot_correlation is None
    assert list(constant.adjust([1, 3], [7, 9])) == [7, 9]
    assignment = np.repeat([0, 1], 4)
    covariate = np.tile(pilot_x, 2)
    noise = np.tile([1, -1, -1, 1], 2)
    outcomes = 10 + 3 * covariate + 2 * assignment + noise
    result = analyze_cuped(assignment, outcomes, covariate, fit)
    assert result.unadjusted.absolute_lift == result.adjusted.absolute_lift == 2
    assert result.adjusted.effect_se < result.unadjusted.effect_se
    assert result.variance_reduction == pytest.approx(45 / 46)
