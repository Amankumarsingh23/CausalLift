import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

from causallift.planning import (
    binary_mde,
    binary_power,
    binary_power_warnings,
    binary_sample_size,
    continuous_power,
    continuous_sample_size,
)


@pytest.mark.parametrize("ratio", [1, 0.4, 0.85 / 0.15])
def test_binary_sample_size_matches_statsmodels(ratio):
    baseline, alternative, alpha, target = 0.10, 0.12, 0.05, 0.9
    plan = binary_sample_size(baseline, alternative, ratio=ratio, alpha=alpha, target_power=target)
    h = proportion_effectsize(alternative, baseline)
    ref = NormalIndPower()
    control_n = ref.solve_power(h, alpha=alpha, power=target, ratio=ratio)
    assert 0 <= plan.control_n - control_n <= 1.001
    assert 0 <= plan.treated_n - control_n * ratio <= 1.001
    assert plan.achieved_power >= target
    assert plan.achieved_power == pytest.approx(ref.power(h, plan.control_n, alpha, plan.treated_n / plan.control_n))


def test_binary_power_mde_and_size_round_trip():
    nc, nt = 30_000, 170_000
    for direction in ("increase", "decrease"):
        mde = binary_mde(0.01, nc, nt, direction=direction)
        assert (mde.absolute_effect > 0) == (direction == "increase")
        assert mde.relative_effect == pytest.approx(mde.absolute_effect / 0.01)
        assert binary_power(0.01, mde.alternative, nc, nt) == pytest.approx(0.8)
        plan = binary_sample_size(0.01, mde.alternative, ratio=nt / nc)
        assert abs(plan.control_n - nc) <= 1
        assert abs(plan.treated_n - nt) <= 1
    assert binary_power(0.1, 0.12, 5000, 4000) > binary_power(0.1, 0.12, 500, 400)
    assert binary_sample_size(0.1, absolute_effect=0.01).total_n > binary_sample_size(0.1, 0.12).total_n
    assert binary_sample_size(0.1, relative_effect=0.2).total_n == binary_sample_size(0.1, 0.12).total_n


def test_continuous_planning_matches_normal_reference():
    plan = continuous_sample_size(-0.2, ratio=2, target_power=0.9)
    equivalent = NormalIndPower().power(0.2, plan.control_n, 0.05, plan.treated_n / plan.control_n)
    assert continuous_power(-0.2, plan.control_n, plan.treated_n) == pytest.approx(equivalent)
    assert plan.achieved_power >= 0.9


@pytest.mark.parametrize("call", [
    lambda: binary_sample_size(0, 0.2),
    lambda: binary_sample_size(0.2, 0.2),
    lambda: binary_sample_size(0.2, relative_effect=-2),
    lambda: binary_sample_size(0.2, 0.3, relative_effect=0.2),
    lambda: binary_sample_size(0.2, 0.3, ratio=0),
    lambda: binary_sample_size(0.2, 0.3, alpha=1),
    lambda: binary_sample_size(0.2, 0.3, target_power=0.01),
    lambda: binary_power(0.2, 0.3, 0, 10),
    lambda: binary_mde(0.01, 10, 10, direction="decrease"),
    lambda: continuous_sample_size(1e-200),
])
def test_invalid_or_infeasible_designs(call):
    with pytest.raises(ValueError):
        call()


def test_rare_counts_warn_and_null_power_equals_alpha():
    assert binary_sample_size(0.001, 1).warnings
    assert binary_power_warnings(0.0001, 0.0002, 100, 100)
    assert binary_power(0.004, 0.004, 321, 100) == pytest.approx(0.05)
