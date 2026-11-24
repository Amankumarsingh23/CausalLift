import json

import pandas as pd

from causallift.design_simulation import cuped_examples, run_simulations
from causallift.session01 import persist
from causallift.session03 import run
from causallift.simulation import simulate
from tests.test_data import FIXTURE, FIXTURE_PATH


def test_seeded_simulations_repeat_and_measure_nominal_behavior():
    first = run_simulations(seed=21, repetitions=400)
    assert first == run_simulations(seed=21, repetitions=400)
    rates = {r["check"]: r for r in first}
    assert rates["naive_peeking"]["estimate"] > rates["fixed_horizon"]["estimate"]
    for key in ("fixed_horizon", "planned_alternative", "newcombe_interval",
                "uncorrected", "bonferroni", "holm", "benjamini_hochberg"):
        row = rates[key]
        # Four theoretical Monte Carlo standard errors plus a small approximation allowance.
        margin = 4 * (row["reference"] * (1 - row["reference"]) / 400) ** 0.5 + 0.008
        assert abs(row["estimate"] - row["reference"]) < margin
    assert rates["benjamini_hochberg"]["metric"] == "FDR"
    _, trials = cuped_examples(42, 100)
    assert trials[0]["estimate"] > 0.65
    assert abs(trials[1]["estimate"]) < 0.12


def test_session03_offline_fixture_reports(tmp_path):
    root = tmp_path
    previous = root / "artifacts/session_02"
    previous.mkdir(parents=True)
    pd.DataFrame({"dataset": ["Full archive"] * 2, "outcome": ["conversion", "visit"],
                  "control_n": [100, 100], "treated_n": [100, 100],
                  "control_events": [2, 12]}).to_csv(previous / "criteo_experiment_results.csv", index=False)
    pd.DataFrame({"dataset": ["fixture"] * 3, "feature": ["f0"] * 3, "subgroup": ["low", "medium", "high"],
                  "partition": ["validation"] * 3, "p_value": [0.01, None, 0.06]}).to_csv(
        previous / "subgroup_analysis.csv", index=False,
    )
    processed = root / "data/processed"
    processed.mkdir(parents=True)
    persist(pd.read_csv(FIXTURE_PATH), FIXTURE, processed / "criteo_v2_1.parquet", 42)
    synthetic = root / "data/synthetic"
    synthetic.mkdir()
    for scenario in ("A", "C"):
        frame, _, spec = simulate(scenario, 500)
        persist(frame, spec, synthetic / f"synthetic_{scenario}.parquet", 42)

    summary = run(root, repetitions=12)
    target = root / "artifacts/session_03"
    assert summary["planning_rows"] == 24
    assert summary["mde_rows"] == 16
    assert summary["srm_statuses"]["criteo_released_archive"] == "not_assessable"
    assert len(summary["simulations"]) == 10
    assert len(list(target.iterdir())) == 11
    assert (target / "covariate_balance.png").read_bytes().startswith(b"\x89PNG")
    assert (target / "statistical_simulation_summary.png").read_bytes().startswith(b"\x89PNG")
    decisions = json.loads((target / "guardrail_decisions.json").read_text(encoding="utf-8"))
    assert {r["decision"] for r in decisions.values()} == {"ship", "do_not_ship", "inconclusive", "investigate"}
    adjustments = pd.read_csv(target / "multiple_testing_results.csv")
    assert adjustments["partition"].eq("validation").all()
    assert adjustments["family_size"].eq(3).all()
    assert adjustments.loc[adjustments["subgroup"].eq("medium"), "adjusted_p"].isna().all()
