from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from causallift.optimization import allocate, discount_profit, economics, policy_totals
from causallift.session11 import run, statuses


def exhaustive(profit, cost, elig, budget, capacity):
    return max(
        sum(profit[i] for i in s)
        for k in range(capacity + 1)
        for s in combinations(np.flatnonzero(elig), k)
        if sum(cost[i] for i in s) <= budget
    )


def test_economics_discount_break_even_and_zero_spend():
    result = economics(
        [1, 2, 3], [0.1, 0, 0.2], [40, 0, 30], [2, 0, 0], [True, True, False]
    )
    assert result.profit.tolist() == [2, 0, 6]
    assert np.isnan(result.break_even.iloc[1])
    assert result.break_even.iloc[0] == 0.05
    assert policy_totals([0.1], [40], [0], [0])["roi"] is None
    assert discount_profit(0.7, 0.8, 40, 8, 1) == pytest.approx(-3.4)
    with pytest.raises(ValueError, match="Unique"):
        economics([1, 1], [1, 1], [1, 1], [1, 1], [True, True])


@pytest.mark.parametrize("policy", ("uniform", "milp"))
def test_exact_engines_match_enumeration(policy):
    profit = [5, 5, 9, -1, 3, 2]
    cost = [2] * 6 if policy == "uniform" else [6, 4, 7, 2, 2, 0]
    elig = [True, True, False, True, True, True]
    idx, result = allocate(profit, cost, elig, 8, capacity=3, policy=policy)
    assert result["proven_optimal"]
    assert result["objective"] == exhaustive(profit, cost, elig, 8, 3)
    assert 2 not in idx and 3 not in idx
    assert len(idx) <= 3 and result["spend"] <= 8
    assert np.array_equal(
        idx, allocate(profit, cost, elig, 8, capacity=3, policy=policy)[0]
    )


def test_variable_cost_greedy_can_be_suboptimal_and_free_choices():
    p, c, eligible = [11, 6, 6], [6, 5, 5], [True] * 3
    _, greedy = allocate(p, c, eligible, 10, policy="ratio")
    _, optimal = allocate(p, c, eligible, 10, policy="milp")
    assert greedy["objective"] == 11 < optimal["objective"] == 12
    selected, _ = allocate([3, 4, -1], [0, 1, 0], eligible, 0, policy="ratio")
    np.testing.assert_array_equal(selected, [0])
    assert len(allocate([-1], [0], [True], 0, policy="uniform")[0]) == 0
    assert len(allocate([10], [0], [True], 0, capacity=0)[0]) == 0
    with pytest.raises(ValueError, match="Capacity"):
        allocate([1], [1], [True], 1, capacity=1.5)


def test_multitreatment_never_repeats_customer():
    groups = np.array([42, 42, 45, 45])
    idx, meta = allocate([9, 10, 3, -1], [1, 1, 1, 1], [True] * 4, 4, groups=groups)
    assert set(idx) == {1, 2}
    assert len(np.unique(groups[idx])) == len(idx)
    assert meta["spend"] == 2


def test_fixture_reproduces_learned_assignments_when_truth_changes(
    tmp_path, monkeypatch
):
    from causallift import session11 as mod

    a, b = tmp_path / "original", tmp_path / "changed"
    summary = run(tmp_path, fixture=True, output=a)
    original = mod._frames

    def shifted(*args, **kwargs):
        spec, train, val, truth = original(*args, **kwargs)
        return spec, train, val, truth.assign(tau_true=-truth.tau_true)

    monkeypatch.setattr(mod, "_frames", shifted)
    run(tmp_path, fixture=True, output=b)
    for path in (a, b):
        data = pd.read_csv(path / "policy_optimization_results.csv")
        f = data[data.scenario.eq("F")]
        assert set(f.business_interpretation) == {"research_only"}
        assert not f.deployment_recommended.any()
        assert (data.spend <= data.budget).all()
        assert (data.selected_count <= data.capacity).all()
    first = (
        pd.read_parquet(a / "local_assignments.parquet")
        .query("policy != 'oracle'")
        .reset_index(drop=True)
    )
    second = (
        pd.read_parquet(b / "local_assignments.parquet")
        .query("policy != 'oracle'")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(first, second)
    assert not summary["test_used"]
    assert (
        statuses({"F": {"status": "supported_under_assumptions"}}, "F")[
            "identification"
        ]
        == "not_identified"
    )


def test_provenance_rejects_reserved_or_duplicate_units(tmp_path, monkeypatch):
    from causallift import session11 as mod

    original = mod._frames

    def invalid(*args, **kwargs):
        spec, train, val, truth = original(*args, **kwargs)
        return spec, train, val.assign(split="test"), truth

    monkeypatch.setattr(mod, "_frames", invalid)
    with pytest.raises(ValueError, match="partitions"):
        run(tmp_path, fixture=True)
