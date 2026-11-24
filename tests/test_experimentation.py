from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import fisher_exact, norm, ttest_ind
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from causallift.data.analytics import binary_arm_counts
from causallift.data.criteo import CRITEO, FIXTURE
from causallift.experimentation import (
    analyze_binary,
    analyze_continuous,
    binary_from_counts,
)
from causallift.subgroups import subgroup_analysis
from tests.test_data import FIXTURE_PATH


def test_binary_effect_and_wald_interval():
    result = binary_from_counts(120, 1000, 100, 1000, dataset_id="fixture", experiment_id="conversion")
    assert result.total_n == 2000
    assert result.treatment.events == 120
    assert result.control.mean == 0.10
    assert result.treatment.mean == 0.12
    assert result.absolute_lift == pytest.approx(0.02)
    assert result.relative_lift == pytest.approx(0.20)
    assert result.effect_se == pytest.approx(np.sqrt(0.12 * 0.88 / 1000 + 0.10 * 0.90 / 1000))
    assert result.null_se == pytest.approx(np.sqrt(0.11 * 0.89 * 2 / 1000))
    assert result.wald_interval == pytest.approx(0.02 + np.array([-1, 1]) * norm.ppf(0.975) * result.effect_se)
    assert result.dataset_id == "fixture"
    assert result.experiment_id == "conversion"


def test_pooled_z_matches_statsmodels_and_swaps_sign():
    result = binary_from_counts(525, 170_000, 58, 30_000)
    z, p = proportions_ztest([525, 58], [170_000, 30_000])
    assert result.test_method == "pooled_proportion_z"
    assert (result.statistic, result.p_value) == pytest.approx((z, p))
    assert result.significant
    swapped = binary_from_counts(58, 30_000, 525, 170_000)
    assert swapped.absolute_lift == -result.absolute_lift
    assert swapped.statistic == -result.statistic
    assert swapped.p_value == result.p_value
    assert swapped.confidence_interval == pytest.approx(tuple(-x for x in reversed(result.confidence_interval)))


@pytest.mark.parametrize(("xt", "nt", "xc", "nc", "level"), [
    (1, 50, 0, 35, 0.95), (55, 800, 38, 1100, 0.90), (12, 12, 0, 9, 0.95),
])
def test_newcombe_matches_statsmodels_including_rare_events(xt, nt, xc, nc, level):
    result = binary_from_counts(xt, nt, xc, nc, confidence_level=level)
    expected = confint_proportions_2indep(xt, nt, xc, nc, compare="diff", method="newcomb", alpha=1 - level)
    assert result.confidence_interval == pytest.approx(expected, abs=1e-14)
    assert result.interval_method == "newcombe_wilson"
    assert -1 <= result.confidence_interval[0] <= result.confidence_interval[1] <= 1


def test_sparse_fisher_and_undefined_relative_lift():
    result = binary_from_counts(3, 15, 0, 11)
    assert result.test_method == "fisher_exact"
    assert result.statistic_name == "sample_odds_ratio"
    assert result.statistic is None
    assert result.p_value == pytest.approx(fisher_exact([[3, 12], [0, 11]]).pvalue)
    assert result.absolute_lift == pytest.approx(0.2)
    assert result.relative_lift is None
    assert any("Relative lift is undefined" in warning for warning in result.warnings)


@pytest.mark.parametrize("event", [0, 1])
def test_constant_binary_outcomes(event):
    result = binary_from_counts(event * 10, 10, event * 16, 16, bootstrap_resamples=20)
    assert result.absolute_lift == 0
    assert result.p_value == 1
    assert not result.significant
    assert result.effect_se == result.null_se == 0
    lo, hi = result.confidence_interval
    assert lo < 0 < hi
    assert result.wald_interval == result.bootstrap_interval == (0, 0)
    assert any("collapses" in warning for warning in result.warnings)


def test_large_z_log_probability_remains_available():
    result = binary_from_counts(576824, 11882655, 80105, 2096937)
    assert result.p_value == 0
    assert result.log10_p_value == pytest.approx(-926.3578364628)
    assert result.significant
    assert any("underflowed" in warning for warning in result.warnings)


def test_welch_matches_scipy_with_different_sizes_and_variances():
    control = np.array([2, 3, 3, 4, 5, 5])
    treated = np.array([1, 3, 7, 12])
    result = analyze_continuous(np.repeat([0, 1], [len(control), len(treated)]), np.r_[control, treated])
    expected = ttest_ind(treated, control, equal_var=False)
    assert (result.statistic, result.p_value, result.degrees_of_freedom) == pytest.approx(
        (expected.statistic, expected.pvalue, expected.df),
    )
    assert result.confidence_interval == pytest.approx(expected.confidence_interval())
    assert result.control.variance == pytest.approx(control.var(ddof=1))
    assert result.treatment.variance == pytest.approx(treated.var(ddof=1))
    assert result.absolute_lift == pytest.approx(treated.mean() - control.mean())


def test_zero_variance_welch():
    assignment = [0, 0, 1, 1]
    constant = analyze_continuous(assignment, [0, 0, 1, 1])
    assert constant.test_method == "unavailable"
    assert constant.relative_lift is None
    assert constant.p_value is constant.significant is constant.confidence_interval is None
    single_constant_arm = analyze_continuous(assignment, [2, 2, 6, 10])
    assert single_constant_arm.test_method == "welch_t"
    assert single_constant_arm.degrees_of_freedom == pytest.approx(1)


def test_bootstraps_match_independent_arm_resampling():
    seed, reps = 12, 500
    result = binary_from_counts(4, 10, 3, 9, seed=seed, bootstrap_resamples=reps)
    rng = np.random.default_rng(seed)
    binary = rng.binomial(10, 0.4, reps) / 10 - rng.binomial(9, 1 / 3, reps) / 9
    assert result.bootstrap_interval == pytest.approx(np.quantile(binary, [0.025, 0.975]))
    control, treated = [1, 4, 6], [6, 9]
    continuous = analyze_continuous([0, 0, 0, 1, 1], control + treated, seed=seed, bootstrap_resamples=reps)
    rng = np.random.default_rng(seed)
    draws = [rng.choice(treated, 2).mean() - rng.choice(control, 3).mean() for _ in range(reps)]
    assert continuous.bootstrap_interval == pytest.approx(np.quantile(draws, [0.025, 0.975]))


@pytest.mark.parametrize(("assignment", "outcome", "message"), [
    ([0, 3], [0, 1], "assignment"), ([0, 1], [0, 2], "Binary outcomes"),
    ([0, 1], [0, np.nan], "finite"), ([0, 0], [0, 1], "nonempty"),
    ([], [], "nonempty"), ([0], [0, 1], "equally sized"), ([0, 1], [0, "1"], "numeric"),
])
def test_binary_invalid_rows(assignment, outcome, message):
    with pytest.raises(ValueError, match=message):
        analyze_binary(assignment, outcome)


def test_invalid_counts_and_continuous_samples():
    with pytest.raises(ValueError, match="integers"):
        binary_from_counts(1.5, 5, 2, 8)
    with pytest.raises(ValueError, match="between zero and arm size"):
        binary_from_counts(6, 5, 2, 8)
    with pytest.raises(ValueError, match="Confidence level"):
        binary_from_counts(1, 4, 1, 4, confidence_level=1)
    with pytest.raises(ValueError, match="at least two observations"):
        analyze_continuous([0, 0, 1], [1, 2, 3])
    with pytest.raises(ValueError, match="finite"):
        analyze_continuous([0, 0, 1, 1], [0, 1, 2, np.inf])


def test_exact_sql_totals_match_rows_for_both_formats(tmp_path):
    fixture = pd.read_csv(FIXTURE_PATH)
    reordered_csv = tmp_path / "fixture.csv"
    fixture[sorted(fixture.columns)].to_csv(reordered_csv, index=False)
    parquet = tmp_path / "fixture.parquet"
    fixture.to_parquet(parquet, index=False)
    for path in (reordered_csv, parquet):
        control, treated = binary_arm_counts(path, FIXTURE, csv=path == reordered_csv)
        for index, outcome in enumerate(("conversion", "visit"), start=1):
            from_counts = binary_from_counts(treated[index], treated[0], control[index], control[0])
            assert from_counts == analyze_binary(fixture["treatment"], fixture[outcome])
    fixture.loc[0, "visit"] = 2
    fixture.to_parquet(parquet, index=False)
    with pytest.raises(ValueError, match="invalid outcomes"):
        binary_arm_counts(parquet, FIXTURE)
    fixture["treatment"] = 0
    fixture.to_parquet(parquet, index=False)
    with pytest.raises(ValueError, match="both arms"):
        binary_arm_counts(parquet, FIXTURE)


def test_subgroups_report_counts_and_unavailable_contrasts():
    frame = pd.DataFrame({"f0": ["low", "low", "low", "low", "high"],
                          "treatment": [1, 1, 0, 0, 1], "conversion": [1, 1, 1, 0, 0]})
    spec = replace(CRITEO, feature_columns=("f0",))
    high, low = subgroup_analysis(frame, spec, "f0")
    assert low["exploratory"]
    assert (low["treated_n"], low["control_n"], low["treated_events"], low["control_events"]) == (2, 2, 2, 1)
    assert (low["treated_rate"], low["control_rate"], low["absolute_lift"]) == (1, 0.5, 0.5)
    assert high["absolute_lift"] is high["p_value"] is high["control_rate"] is None
    assert "Insufficient data" in high["warnings"]
    with pytest.raises(ValueError, match="allowlisted"):
        subgroup_analysis(frame, spec, "exposure")
    frame["f0"] = [0, -1, -0.5, -1, 2]
    bins = subgroup_analysis(frame, spec, "f0", bins=(-np.inf, 0, 1, np.inf))
    assert bins[0]["absolute_lift"] == 0.5
    assert (bins[1]["treated_n"], bins[1]["control_n"]) == (0, 0)
    assert bins[1]["ci_low"] is None
