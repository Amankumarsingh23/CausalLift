import json

import numpy as np
import pandas as pd
import pytest

from causallift.observational import aipw
from causallift.policy_evaluation import PolicyEvaluation
from causallift.session12 import _action_hash, recover_action, run, simulation


def test_paired_scores_apply_unit_contributions_and_exact_costs():
    evaluator = PolicyEvaluation(
        range(4),
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        0.5,
        0.2,
        0.4,
        contribution=[10, 20, 10, 20],
        cost=[1, 3, 1, 3],
    )
    rows = {row["method"]: row for row in evaluator.compare([1, 1, 0, 0])}
    assert rows["dm"]["incremental_profit"] == pytest.approx(2)
    assert rows["dm"]["value_rate"] == pytest.approx(0.3)
    assert rows["ips"]["value_rate"] == pytest.approx(0.5)
    assert rows["snips"]["value_rate"] == pytest.approx(0.5)
    assert rows["ips"]["incremental_profit"] == pytest.approx(16)
    assert rows["dr"]["value_rate"] == pytest.approx(0.5)
    assert rows["dr"]["incremental_profit"] == pytest.approx(10)
    assert rows["dr"]["roi"] == pytest.approx(2.5)
    assert rows["dr"]["spend"] == 4
    assert rows["ips"]["matched_n"] == 2
    assert rows["ips"]["ess"] == 2
    assert rows["ips"]["max_weight"] == 2
    assert rows["snips"]["incremental_profit"] is None
    for row in evaluator.compare([1, 1, 0, 0], [1, 1, 0, 0]):
        assert row["incremental_rate"] == 0
    for row in evaluator.compare([0, 0, 0, 0]):
        assert row["incremental_rate"] == 0
        assert row["roi"] is None


def test_missing_support_and_invalid_inputs_are_visible():
    evaluator = PolicyEvaluation(range(2), [0, 1], [1, 0], 0.5, 0.2, 0.4)
    rows = {row["method"]: row for row in evaluator.compare([1, 0])}
    assert rows["ips"]["value_rate"] is None
    assert rows["snips"]["value_rate"] is None
    assert rows["dr"]["missing_action_support"]
    assert "ci_rate_low" not in rows["dr"]
    with pytest.raises(ValueError, match="binary"):
        evaluator.compare([np.nan, 0])
    with pytest.raises(ValueError, match="interior"):
        PolicyEvaluation(range(2), [0, 1], [1, 0], 0, 0.2, 0.4)


def test_simulation_is_seeded_and_covers_both_logging_designs():
    first = simulation(42, 2)
    assert first == simulation(42, 2)
    assert {row["propensity"] for row in first} == {0.15, 0.5}


def test_dr_treat_all_minus_none_agrees_with_aipw_ate():
    t, y = [0, 1, 0, 1], [1, 1, 0, 0]
    e = [0.2, 0.4, 0.6, 0.8]
    mu0, mu1 = [0.2] * 4, [0.4] * 4
    evaluator = PolicyEvaluation(range(4), t, y, e, mu0, mu1)
    dr = next(r for r in evaluator.compare([1] * 4) if r["method"] == "dr")
    ate = aipw(t, y, e, mu0, mu1)
    assert dr["incremental_rate"] == pytest.approx(ate["estimate"])
    assert dr["se_rate"] == pytest.approx(ate["se"])


def test_recover_action_checks_identity_eligibility_budget_and_order():
    units = np.array([4, 7])
    value, cost = np.array([5, 6]), np.array([2, 3])
    chosen = pd.DataFrame({"unit_id": [7], "value": [6], "cost": [3]})
    eligible = np.array([True, True])
    a = recover_action(units, chosen, value, cost, eligible, 3, 1, 1, 3)
    assert a.tolist() == [False, True]
    assert _action_hash(units, a) != _action_hash(units[::-1], a[::-1])
    with pytest.raises(ValueError, match="identities"):
        recover_action(
            units, chosen.assign(unit_id=8), value, cost, eligible, 3, 1, 1, 3
        )
    with pytest.raises(ValueError, match="budget"):
        recover_action(units, chosen, value, cost, eligible, 2, 1, 1, 3)
    eligible[1] = False
    with pytest.raises(ValueError, match="eligibility"):
        recover_action(units, chosen, value, cost, eligible, 3, 1, 1, 3)


def test_offline_fixture_keeps_test_locked_and_f_research_only(tmp_path):
    output = tmp_path / "results"
    summary = run(tmp_path, fixture=True, output=output, reps=2)
    assert summary["synthetic_rows"] > 0
    assert summary["criteo_rows"] == 0
    assert summary["test_used"] is False
    assert not list(tmp_path.rglob("*test*.parquet"))
    frozen = json.loads((output / "final_evaluation_protocol.json").read_text())
    assert frozen["test_evaluated"] is False
    assert "pre-treatment" in frozen["test_action_generation"]
    rows = pd.read_csv(output / "synthetic_policy_evaluation.csv")
    f = rows.loc[rows.dataset.eq("F")]
    assert not f.empty
    assert set(f.identification) == {"not_identified"}
    assert f.ci_profit_low.isna().all()
    assert f.true_profit.isna().all()
