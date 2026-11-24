import numpy as np
import pandas as pd
import pytest

from causallift.simulation import simulate
from causallift.targeting import (
    describe_targeting,
    evaluate_synthetic_targeting,
    policy_rankings,
    rank_scores,
)


def test_policy_capacity_ties_and_determinism():
    frame, _, spec = simulate("E", 70)
    train = frame.iloc[:47].assign(split="train")
    validation = frame.iloc[47:].assign(split="validation")
    p = np.linspace(0, 1, len(validation))
    rankings, effect = policy_rankings(p, train, spec, seed=123)
    again, _ = policy_rankings(p, train, spec, seed=123)
    np.testing.assert_array_equal(again["random"], rankings["random"])
    np.testing.assert_array_equal(rankings["constant_effect"], rankings["random"])
    np.testing.assert_array_equal(rankings["predictive"][:3], [22, 21, 20])
    np.testing.assert_array_equal(np.sort(rank_scores([1, 1, 0])[:2]), [0, 1])
    assert effect == pytest.approx(train.loc[train.treatment.eq(1), "outcome"].mean()
                                  - train.loc[train.treatment.eq(0), "outcome"].mean())
    rows = describe_targeting(rankings, p, validation, spec)
    for fraction, n in ((0.1, 2), (0.2, 4), (0.5, 11)):
        assert {row["selected_n"] for row in rows if row["fraction"] == fraction} == {n}
    assert all(row["mean_true_cate"] is None for row in rows)
    with pytest.raises(ValueError, match="exactly once"):
        describe_targeting({"predictive": np.zeros(len(p), dtype=int)}, p, validation, spec)


def test_synthetic_oracle_joins_truth_in_validation_order(tmp_path):
    frame, _, spec = simulate("E", 60)
    train = frame.iloc[:40].assign(split="train")
    validation = frame.iloc[40:].assign(split="validation")
    p = np.arange(20) / 20
    tau = np.r_[[0.2] * 10, [-0.1] * 10]
    truth = pd.DataFrame({"unit_id": validation.unit_id, "tau_true": tau})
    path = tmp_path / "truth.parquet"
    truth.sample(frac=1, random_state=9).to_parquet(path, index=False)
    rankings = policy_rankings(p, train, spec)[0]
    results = evaluate_synthetic_targeting(rankings, p, validation, spec, path)
    selected = {row["policy"]: row for row in results if row["fraction"] == 0.2}
    assert {row["selected_n"] for row in selected.values()} == {4}
    assert selected["predictive"]["mean_true_cate"] == pytest.approx(-0.1)
    assert selected["predictive"]["expected_incremental_conversions"] == pytest.approx(-0.4)
    assert selected["predictive"]["negative_effect_fraction"] == 1
    assert selected["oracle"]["expected_incremental_conversions"] == pytest.approx(0.8)
    assert selected["oracle"]["negative_effect_fraction"] == 0
    truth.iloc[:-1].to_parquet(path, index=False)
    with pytest.raises(ValueError, match="truth is missing"):
        evaluate_synthetic_targeting(rankings, p, validation, spec, path)
