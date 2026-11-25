"""Evaluate frozen policy actions on development data and lock the final protocol."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

from causallift.data.criteo import CRITEO
from causallift.meta_learners import MetaLearner
from causallift.optimization import allocate
from causallift.policy_evaluation import INTERVAL_NOTE, PolicyEvaluation
from causallift.prediction import training_fingerprint
from causallift.session01 import write_json
from causallift.session04 import _partitions
from causallift.session05 import _predictive_baseline
from causallift.session06 import _manifest
from causallift.session07 import _fingerprint, _load_previous
from causallift.session09 import load_saved_propensity
from causallift.session11 import SETTINGS, _design, _frames
from causallift.session11 import run as allocate_fixture
from causallift.targeting import aligned_synthetic_effects

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

POLICIES = (
    "none",
    "random",
    "predictive",
    "uplift",
    "profit",
    "profit_to_cost",
    "integer",
    "oracle",
)
ECONOMIC_SEED = 42


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _action_hash(units, action):
    rec = np.column_stack([units, action]).astype("<i8")
    return hashlib.sha256(rec.tobytes()).hexdigest()


def recover_action(
    units, chosen, value, cost, eligible, budget, capacity, count, spend
):
    ids = pd.Index(units)
    keys = pd.Index(chosen.unit_id)
    if not ids.is_unique or not keys.is_unique or not keys.isin(ids).all():
        raise ValueError(
            "Assignment unit identities are duplicate or outside the cohort"
        )
    i = ids.get_indexer(keys)
    a = np.zeros(len(ids), dtype=bool)
    a[i] = True
    if (
        not eligible[i].all()
        or not np.array_equal(chosen.value, value[i])
        or not np.array_equal(chosen.cost, cost[i])
        or len(i) != count
        or count > capacity
        or not np.isclose(cost[i].sum(), spend, rtol=0, atol=1e-8)
        or spend > budget + 1e-8
    ):
        raise ValueError("Assignment budget, capacity, eligibility or economics differ")
    return a


def simulation(seed, reps):
    rng = np.random.default_rng(seed)
    rows = []
    for p in (0.50, 0.15):
        for rep in range(reps):
            x = rng.uniform(-1, 1, 1600)
            mu0 = 0.22 + 0.08 * x
            tau = 0.04 + 0.13 * x
            t = rng.binomial(1, p, len(x))
            y = rng.binomial(1, mu0 + t * tau)
            a = x > 0.4
            # Prespecified imperfect outcome models, independent of the logged outcomes.
            evaluator = PolicyEvaluation(
                np.arange(len(x)), t, y, p, 0.22, 0.25, contribution=45, cost=2
            )
            truth = float((tau * a).mean())
            true_profit = float(((45 * tau - 2) * a).sum())
            for r in evaluator.compare(a):
                if r["method"] == "snips":
                    continue
                delta = r["incremental_rate"]
                rows.append(
                    {
                        "propensity": p,
                        "rep": rep,
                        "method": r["method"],
                        "true_rate": truth,
                        "error": delta - truth,
                        "estimate": delta,
                        "true_profit": true_profit,
                        "estimated_profit": r["incremental_profit"],
                        "ess": r["ess"],
                        "match_fraction": r["match_fraction"],
                        "covered": r["ci_rate_low"] <= truth <= r["ci_rate_high"],
                    }
                )
    return rows


def protocol():
    return {
        "protocol": "CausalLift final evaluation v1",
        "reserved_split": "test",
        "test_evaluated": False,
        "seed": 42,
        "models": {
            "policy_effect": "Session 05 S-LightGBM",
            "outcome_nuisance": "Session 05 T-logistic",
            "predictive": "Session 04 pooled LightGBM",
        },
        "primary_cohorts": ["synthetic_E", "Criteo 1M"],
        "secondary_cohorts": [
            "synthetic_A",
            "synthetic_B",
            "synthetic_D",
            "Criteo 200k",
        ],
        "primary_outcome": "conversion or synthetic binary outcome",
        "policy_reference": "treat none",
        "primary_comparison": "profit versus predictive at 20% nominal capacity",
        "primary_metrics": [
            "paired DR incremental profit",
            "paired incremental conversions",
            "ESS",
            "synthetic true expected incremental profit",
        ],
        "secondary_metrics": [
            "DM",
            "IPS",
            "SNIPS outcome only",
            "DR",
            "proposed spend",
            "ROI versus no treatment",
        ],
        "synthetic_economics": {
            "contribution": 45,
            "cost": 2,
            "eligible_probability": 0.9,
            "eligibility_seed_components": [42, "ASCII of scenario", 0],
            "capacity_fraction": 0.30,
            "budget_ceiling": "floor(2 × full test population × fraction)",
            "fractions": [0.05, 0.10, 0.20],
        },
        "criteo_economics": {
            "contribution": "synthetic integer draws 60–140 inclusive",
            "cost": 0.4,
            "eligibility": "all",
            "fractions": [0.10, 0.20, 0.50],
            "budget": "cost × floor(fraction × test count)",
        },
        "design_propensities": {
            "synthetic": "known assignment probability",
            "criteo": "released test assignment share, approximation only",
        },
        "policy_rules": {
            "none": "all zero",
            "random": "seeded rank then scan affordable eligible",
            "predictive": "descending pooled score then affordable eligible scan",
            "uplift": "descending S-LightGBM CATE then scan",
            "profit": "exact positive estimated-profit top-K under uniform cost",
            "ties": "Session 11 seeded stable ranking",
        },
        "test_action_generation": "Load unchanged saved models and verify training provenance. Use only ordered test IDs and allowlisted pre-treatment covariates plus frozen economics and counts for action construction. Freeze vectors before joining test assignment, outcomes or synthetic truth. No fitting or selection after test access.",
        "uncertainty": INTERVAL_NOTE
        + " Report paired outcome and profit intervals, matched counts and ESS. Suppress intervals when matching support is absent.",
        "research_only": "F excluded from primary comparisons. Criteo is a hypothetical released-sample diagnostic.",
    }


def _record(results, support, paired, evaluator, actions, keys, truth, intervals):
    for policy, a in actions.items():
        policy_key = keys | {
            "policy": policy,
            "allocation_hash": _action_hash(evaluator.units, a),
        }
        rows = evaluator.compare(a, intervals=intervals)
        known = (
            None
            if truth is None
            else float((truth[a] * evaluator.value[a] - evaluator.cost[a]).sum())
        )
        for row in rows:
            results.append(
                policy_key
                | {
                    "policy": policy,
                    "true_conversions": None
                    if truth is None
                    else float(truth[a].sum()),
                    "true_profit": known,
                    "profit_error": None
                    if known is None or row["incremental_profit"] is None
                    else row["incremental_profit"] - known,
                    **row,
                }
            )
        support.append(policy_key | evaluator.support(a, evaluator.scores(a)[1]))
    if "predictive" in actions and "profit" in actions:
        a, b = actions["profit"], actions["predictive"]
        true_delta = (
            None
            if truth is None
            else float(
                np.sum(
                    (a.astype(int) - b.astype(int))
                    * (truth * evaluator.value - evaluator.cost)
                )
            )
        )
        paired.extend(
            keys
            | {
                "policy": "profit",
                "reference": "predictive",
                "true_profit_difference": true_delta,
                "allocation_hash": _action_hash(evaluator.units, a),
                "reference_allocation_hash": _action_hash(evaluator.units, b),
                **row,
            }
            for row in evaluator.compare(
                actions["profit"], actions["predictive"], intervals=intervals
            )
        )


def run(root, *, fixture=False, output=None, seed=42, reps=80):
    if reps < 1:
        raise ValueError("Repetitions must be positive")
    start = perf_counter()
    target = output or root / "artifacts" / (
        "session_12_fixture" if fixture else "session_12"
    )
    target.mkdir(parents=True, exist_ok=True)
    prior_dir = root / "artifacts/session_11"
    if fixture:
        prior_dir = target / "local_fixture_inputs"
        allocate_fixture(root, fixture=True, output=prior_dir, seed=ECONOMIC_SEED)
    files = [
        "policy_reliability_summary.json",
        "policy_optimization_results.csv",
        "local_assignments.parquet",
        "economic_scenarios.json",
    ]
    provenance = json.loads((prior_dir / files[0]).read_text(encoding="utf-8"))
    recorded = pd.read_csv(prior_dir / files[1])
    allocated = pd.read_parquet(prior_dir / files[2])
    assumptions = json.loads((prior_dir / files[3]).read_text(encoding="utf-8"))
    if assumptions["scenarios"] != SETTINGS or assumptions["seed"] != ECONOMIC_SEED:
        raise ValueError("Saved economic scenarios differ from frozen assumptions")
    groupkeys = ["scenario", "economy", "budget", "policy"]
    if (
        recorded.duplicated(groupkeys).any()
        or allocated.duplicated([*groupkeys, "unit_id"]).any()
    ):
        raise ValueError("Duplicate saved policy rows or selected unit IDs")
    valid_keys = set(map(tuple, recorded[groupkeys].to_numpy()))
    if not set(map(tuple, allocated[groupkeys].to_numpy())) <= valid_keys:
        raise ValueError("Unexpected allocation key")
    st = (
        {}
        if fixture
        else _manifest(root / "artifacts/session_05/causal_model_manifest.json")
    )
    pm = (
        {}
        if fixture
        else {
            key[:2]: entry
            for key, entry in _manifest(
                root / "artifacts/session_04/predictive_model_manifest.json",
                "model",
                "lightgbm",
            ).items()
        }
    )
    synthetic, criteo, support, paired, cost_rows, local = [], [], [], [], [], []
    for scenario in "ABDEF":
        spec, train, val, truth_source = _frames(root, scenario, fixture, ECONOMIC_SEED)
        keys = provenance[scenario]
        if (
            keys["split"] != "validation"
            or not train.split.eq("train").all()
            or not val.split.eq("validation").all()
            or tuple(keys["feature_allowlist"]) != tuple(spec.feature_columns)
            or keys["train_fingerprint"]
            != training_fingerprint(train, spec, "treatment")
            or keys["validation_fingerprint"]
            != training_fingerprint(val, spec, "treatment")
            or not val.unit_id.is_unique
            or not train.unit_id.is_unique
            or set(train.unit_id) & set(val.unit_id)
        ):
            raise ValueError("Saved policy provenance or validation rows differ")
        x = spec.features(val)
        if fixture or scenario == "F":
            nuisance = MetaLearner("t", "logistic", seed=ECONOMIC_SEED).fit(
                spec.features(train), train.treatment, train.outcome
            )
        else:
            fingerprint = _fingerprint(train, spec, "outcome")
            nuisance = _load_previous(
                root,
                st,
                (f"synthetic_{scenario}", "outcome", "t_logistic"),
                fingerprint,
                spec.feature_columns,
                "05",
            )[0]
        pot = nuisance.predict_potential_outcomes(x)
        truth = None
        if scenario != "F":
            truth = (
                truth_source.set_index("unit_id")
                .loc[val.unit_id, "tau_true"]
                .to_numpy()
                if fixture
                else aligned_synthetic_effects(val, spec, truth_source)
            )
            e = spec.assignment_probability
        elif fixture:
            e = np.full(len(val), 0.5)
        else:
            _, e, _ = load_saved_propensity(
                root / "artifacts/session_08", scenario, spec, train, val, ECONOMIC_SEED
            )
        for j, economy in enumerate(SETTINGS if scenario != "F" else ("uniform",)):
            rng = np.random.default_rng(
                np.random.SeedSequence([ECONOMIC_SEED, ord(scenario), j])
            )
            n = min(180, len(val)) if economy == "variable_cost" else len(val)
            idx = (
                np.sort(rng.choice(len(val), n, replace=False))
                if n < len(val)
                else np.arange(n)
            )
            value, cost, eligible = _design(economy, n, rng)
            frame = val.iloc[idx]
            units = frame.unit_id.to_numpy()
            cohort_truth = None if truth is None else truth[idx]
            ev = PolicyEvaluation(
                units,
                frame.treatment,
                frame.outcome,
                np.broadcast_to(e, (len(val),))[idx],
                pot.mu0[idx],
                pot.mu1[idx],
                contribution=value,
                cost=cost,
            )
            for budget, g in recorded.loc[
                recorded.scenario.eq(scenario) & recorded.economy.eq(economy)
            ].groupby("budget"):
                actions = {}
                for row in g.itertuples():
                    if row.policy not in POLICIES:
                        continue
                    if row.population_n != n or row.capacity != int(0.30 * n):
                        raise ValueError("Saved population or capacity differs")
                    chosen = allocated.loc[
                        allocated.scenario.eq(scenario)
                        & allocated.economy.eq(economy)
                        & allocated.budget.eq(budget)
                        & allocated.policy.eq(row.policy)
                    ]
                    a = recover_action(
                        units,
                        chosen,
                        value,
                        cost,
                        eligible,
                        budget,
                        row.capacity,
                        row.selected_count,
                        row.spend,
                    )
                    if cohort_truth is not None:
                        np.testing.assert_allclose(
                            np.sum((cohort_truth * value - cost)[a]),
                            row.true_profit,
                            atol=1e-8,
                        )
                    actions[row.policy] = a
                    local.extend(
                        {
                            "dataset": scenario,
                            "economy": economy,
                            "budget": budget,
                            "policy": row.policy,
                            "unit_id": int(i),
                        }
                        for i in units[a]
                    )
                if "none" not in actions or actions["none"].any():
                    raise ValueError("Missing or invalid treat-none reference")
                _record(
                    synthetic,
                    support,
                    paired,
                    ev,
                    actions,
                    {
                        "dataset": scenario,
                        "economy": economy,
                        "budget": int(budget),
                        "population": "full comparison cohort including ineligible",
                        "identification": keys["identification"],
                        "interpretation": keys["business_interpretation"],
                        "logging_propensity_source": "known randomized design"
                        if scenario != "F"
                        else "observed-feature diagnostic",
                        "outcome_nuisance": "T-logistic from Session 05"
                        if not fixture and scenario != "F"
                        else "training-only T-logistic",
                        "eligible_n": int(eligible.sum()),
                        "capacity": int(0.30 * n),
                    },
                    cohort_truth,
                    scenario != "F",
                )
            if scenario == "E" and economy == "uniform":
                g = recorded.loc[
                    recorded.scenario.eq(scenario) & recorded.economy.eq(economy)
                ]
                top = int(g.budget.max())
                if fixture:
                    effect = 0.08 * np.tanh(x.engagement.to_numpy())
                    pp = (np.tanh(x.loyalty.to_numpy()) + 1) / 2
                else:
                    effect = _load_previous(
                        root,
                        st,
                        ("synthetic_E", "outcome", "s_lightgbm"),
                        _fingerprint(train, spec, "outcome"),
                        spec.feature_columns,
                        "05",
                    )[0].predict_effect(x)
                    pp = _predictive_baseline(
                        root,
                        False,
                        "synthetic_E",
                        spec,
                        train,
                        val,
                        "outcome",
                        seed,
                        pm,
                    )
                for c in (1, 2, 4):
                    for v in (30, 45, 60):
                        for policy, scores in (
                            ("random", np.zeros(n)),
                            ("predictive", pp),
                            ("uplift", effect),
                            ("profit", None),
                            ("oracle", None),
                        ):
                            decision = cohort_truth if policy == "oracle" else effect
                            chosen, meta = allocate(
                                decision * v - c,
                                np.full(n, c),
                                eligible,
                                top,
                                capacity=int(0.30 * n),
                                policy="ranking" if scores is not None else "uniform",
                                score=scores,
                            )
                            cost_rows.append(
                                {
                                    "dataset": "E",
                                    "cost": c,
                                    "value": v,
                                    "budget": top,
                                    "policy": policy,
                                    "selected": len(chosen),
                                    "spend": meta["spend"],
                                    "true_profit": float(
                                        (cohort_truth[chosen] * v - c).sum()
                                    ),
                                }
                            )
    if not fixture:
        for name, source in (
            ("Criteo 200k", "data/processed/criteo_v2_1.parquet"),
            (
                "Criteo 1M",
                "data/processed/benchmark_1m/data/processed/criteo_v2_1.parquet",
            ),
        ):
            train, val = _partitions(root / source, CRITEO)
            x = CRITEO.features(val)
            fp = _fingerprint(train, CRITEO, "conversion")
            pot = _load_previous(
                root,
                st,
                (name, "conversion", "t_logistic"),
                fp,
                CRITEO.feature_columns,
                "05",
            )[0].predict_potential_outcomes(x)
            effect = _load_previous(
                root,
                st,
                (name, "conversion", "s_lightgbm"),
                fp,
                CRITEO.feature_columns,
                "05",
            )[0].predict_effect(x)
            pp = _predictive_baseline(
                root, False, name, CRITEO, train, val, "conversion", seed, pm
            )
            n = len(x)
            ids = np.arange(n)
            value = np.random.default_rng(ECONOMIC_SEED).integers(60, 141, n)
            cost = np.full(n, 0.4)
            ev = PolicyEvaluation(
                ids,
                val.treatment,
                val.conversion,
                val.treatment.mean(),
                pot.mu0,
                pot.mu1,
                contribution=value,
                cost=cost,
            )
            for fraction in (0.10, 0.20, 0.50):
                cap = int(n * fraction)
                budget = round(0.4 * cap, 2)
                actions = {"none": np.zeros(n, dtype=bool)}
                for label, score in (
                    ("random", np.zeros(n)),
                    ("predictive", pp),
                    ("uplift", effect),
                    ("profit", None),
                ):
                    i, _ = allocate(
                        effect * value - cost,
                        cost,
                        np.ones(n, dtype=bool),
                        budget,
                        capacity=cap,
                        policy="ranking" if score is not None else "uniform",
                        score=score,
                        seed=seed,
                    )
                    a = np.zeros(n, dtype=bool)
                    a[i] = True
                    actions[label] = a
                    local.extend(
                        {
                            "dataset": name,
                            "economy": "hypothetical",
                            "budget": budget,
                            "policy": label,
                            "unit_id": int(j),
                        }
                        for j in i
                    )
                _record(
                    criteo,
                    support,
                    paired,
                    ev,
                    actions,
                    {
                        "dataset": name,
                        "economy": "hypothetical",
                        "budget": budget,
                        "capacity": cap,
                        "identification": "not_assessed",
                        "interpretation": "released-benchmark approximation; hypothetical economics",
                        "population": "full validation",
                        "logging_propensity_source": "released validation assignment share",
                        "logging_propensity": float(val.treatment.mean()),
                        "outcome_nuisance": "Session 05 T-logistic",
                        "positives": int(val.conversion.sum()),
                        "control_positives": int(
                            val.loc[val.treatment.eq(0), "conversion"].sum()
                        ),
                        "outcome_fingerprint": training_fingerprint(
                            val, CRITEO, "conversion"
                        ),
                        "validation_fingerprint": training_fingerprint(
                            val, CRITEO, "treatment"
                        ),
                    },
                    None,
                    True,
                )
    sim = simulation(seed, min(reps, 4) if fixture else reps)
    stats = []
    for (e, m), g in pd.DataFrame(sim).groupby(["propensity", "method"]):
        stats.append(
            {
                "propensity": e,
                "method": m,
                "n_reps": len(g),
                "bias": float(g.error.mean()),
                "rmse": float(np.sqrt(np.mean(g.error**2))),
                "sd": float(g.estimate.std(ddof=1)) if len(g) > 1 else None,
                "coverage": float(g.covered.mean()),
                "ess": float(g.ess.mean()),
                "match_fraction": float(g.match_fraction.mean()),
            }
        )
    frontier = [
        row for row in synthetic if row["dataset"] == "E" and row["method"] == "dr"
    ]
    for name, rows in (
        ("synthetic_policy_evaluation", synthetic),
        ("criteo_policy_evaluation", criteo),
        ("paired_policy_comparisons", paired),
        ("policy_support_diagnostics", support),
        ("policy_simulation_results", sim),
        ("policy_simulation_summary", stats),
        ("budget_profit_frontier", frontier),
        ("cost_sensitivity", cost_rows),
    ):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    pd.DataFrame(local).to_parquet(
        target / "local_policy_assignments.parquet", index=False
    )
    _plots(target, frontier, cost_rows, stats)
    selected = next(
        r["budget"] for r in reversed(frontier) if r["economy"] == "uniform"
    )
    e_summary = [
        {k: r[k] for k in ("policy", "true_profit", "incremental_profit", "spend")}
        for r in frontier
        if r["economy"] == "uniform" and r["budget"] == selected
    ]
    write_json(
        target / "business_decision_summary.json",
        {
            "synthetic_E_uniform_high_budget": e_summary,
            "simulation": stats,
            "limitations": [
                "Coarse conditional intervals exclude reoptimization and trial clustering.",
                "Criteo assignment approximation and hypothetical money.",
                "Scenario F is research-only; no intervals asserted.",
            ],
        },
    )
    frozen = protocol()
    frozen["fixture_only"] = fixture
    frozen["model_artifact_sha256"] = (
        {}
        if fixture
        else {
            f"session_{session}/{entry['model_path']}": _digest(
                root / f"artifacts/session_{session}" / entry["model_path"]
            )
            for session, entries in (("04", pm.values()), ("05", st.values()))
            for entry in entries
            if session == "04" or entry["policy"] in ("s_lightgbm", "t_logistic")
        }
    )
    write_json(target / "final_evaluation_protocol.json", frozen)
    summary = {
        "runtime_seconds": perf_counter() - start,
        "fixture": fixture,
        "seed": seed,
        "test_used": False,
        "synthetic_rows": len(synthetic),
        "criteo_rows": len(criteo),
        "input_sha256": {file: _digest(prior_dir / file) for file in files},
        "model_manifests_sha256": {}
        if fixture
        else {
            n: _digest(root / f"artifacts/session_{n}" / relative)
            for n, relative in (
                ("04", "predictive_model_manifest.json"),
                ("05", "causal_model_manifest.json"),
            )
        },
    }
    write_json(target / "session_12_summary.json", summary)
    return summary


def _plots(target, frontier, cost_rows, stats):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    data = pd.DataFrame(frontier)
    for policy, g in data.loc[data.economy.eq("uniform")].groupby("policy"):
        if policy in ("random", "predictive", "uplift", "profit", "oracle"):
            axes[0].plot(g.budget, g.true_profit, marker="o", label=policy)
    for policy, g in pd.DataFrame(cost_rows).query("value == 45").groupby("policy"):
        axes[1].plot(g.cost, g.true_profit, marker="o", label=policy)
    for ax, label in zip(axes, ("Budget ceiling", "Cost per treatment"), strict=True):
        ax.set(xlabel=label, ylabel="True expected profit (simulated units)")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.legend(fontsize=7)
    fig.savefig(target / "budget_profit_frontier.png", dpi=145)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
    for policy, g in pd.DataFrame(stats).groupby("method"):
        ax.plot(g.propensity, g.rmse * 100, marker="o", label=policy.upper())
    ax.set(xlabel="Logging treatment probability", ylabel="Incremental-rate RMSE (pp)")
    ax.legend()
    fig.savefig(target / "policy_estimator_comparison.png", dpi=145)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reps", type=int, default=80)
    print(run(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
