"""Benchmark profit-aware allocation without touching the reserved test split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.splits import split_data
from causallift.experimentation import analyze_binary
from causallift.optimization import allocate, discount_profit, economics, policy_totals
from causallift.prediction import training_fingerprint
from causallift.session01 import write_json
from causallift.session05 import _predictive_baseline
from causallift.session06 import _manifest
from causallift.session07 import _fingerprint, _load_previous
from causallift.simulation import simulate
from causallift.targeting import aligned_synthetic_effects

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SETTINGS = {
    "uniform": {"value": "45", "cost": "2"},
    "variable_value": {"value": "integer uniform 20–70", "cost": "2"},
    "variable_cost": {"value": "integer uniform 20–70", "cost": "integer uniform 1–7"},
    "unfavorable": {"value": "5", "cost": "20"},
}
FRACTIONS = (0.05, 0.10, 0.20)


def statuses(previous, scenario):
    entry = previous.get(scenario, {})
    status = entry.get(
        "status",
        "supported_under_assumptions" if scenario in "ABDE" else "not_assessed",
    )
    if scenario == "F":
        status = "not_identified"
    interpretation = (
        "research_only" if status == "not_identified" else "simulated_policy_comparison"
    )
    return {
        "identification": status,
        "overlap_status": entry.get("overlap_status", "not_assessed"),
        "business_interpretation": interpretation,
        "deployment_recommended": False,
        "source": "session_10"
        if entry
        else "synthetic randomized design"
        if scenario in "ABDE"
        else "unavailable",
    }


def _frames(root, scenario, fixture, seed):
    if fixture:
        f, g, s = simulate(scenario, 650, seed)
        f["split"] = split_data(f, s, seed)
        return s, f.loc[f.split.eq("train")], f.loc[f.split.eq("validation")], g
    p = root / f"data/synthetic/synthetic_{scenario}.parquet"
    spec = DatasetSpec(
        **json.loads(p.with_suffix(".spec.json").read_text(encoding="utf-8"))
    )
    return (
        spec,
        load_dataset(p, spec, "train"),
        load_dataset(p, spec, "validation"),
        p.with_stem(p.stem + "_truth"),
    )


def _design(name, n, rng):
    value = (
        rng.integers(20, 71, n)
        if name in ("variable_value", "variable_cost")
        else np.full(n, 5 if name == "unfavorable" else 45)
    )
    cost = (
        rng.integers(1, 8, n)
        if name == "variable_cost"
        else np.full(n, 20 if name == "unfavorable" else 2)
    )
    eligible = rng.random(n) >= 0.10
    return value, cost, eligible


def _selections(
    effect,
    value,
    cost,
    eligible,
    budget,
    capacity,
    prediction,
    constant,
    seed,
    variable,
    other,
):
    profit = effect * value - cost
    policies = {
        "none": ("none", None, profit),
        "random": ("ranking", np.zeros(len(effect)), profit),
        "predictive": ("ranking", prediction, profit),
        "constant_effect": ("profit", None, constant * value - cost),
        "uplift": ("ranking", effect, profit),
        "profit": ("profit" if variable else "uniform", None, profit),
    }
    if other is not None:
        policies["uplift_x_linear"] = ("ranking", other, profit)
    if variable:
        policies.update(
            {
                "profit_to_cost": ("ratio", None, profit),
                "integer": ("milp", None, profit),
            }
        )
    return {
        name: allocate(
            p,
            cost,
            eligible,
            budget,
            capacity=capacity,
            policy=policy,
            score=score,
            seed=seed,
        )
        for name, (policy, score, p) in policies.items()
        if score is not None or policy != "ranking"
    }


def _multi(seed):
    rng = np.random.default_rng(seed)
    n = 40
    x = rng.uniform(-1, 1, n)
    mu0 = 0.3 + 0.08 * x
    tau = np.column_stack([0.07 + 0.10 * x, 0.11 - 0.14 * x]).ravel()
    mu1 = mu0[:, None] + tau.reshape(n, 2)
    assert np.all((0 <= mu1) & (mu1 <= 1))
    score = np.column_stack([0.07 + 0.07 * x, 0.10 - 0.08 * x]).ravel()
    value = np.repeat(rng.integers(25, 61, n), 2)
    cost = np.tile([2.0, 3.0], n)
    groups = np.repeat(np.arange(n), 2)
    rows = []
    for name, decision in (("hypothetical_scores", score), ("oracle", tau)):
        selected, solve = allocate(
            decision * value - cost,
            cost,
            np.ones(2 * n, dtype=bool),
            36,
            capacity=15,
            groups=groups,
        )
        rows.append(
            {
                "policy": name,
                "population": n,
                "budget": 36,
                "capacity": 15,
                "action_a": int((selected % 2 == 0).sum()),
                "action_b": int((selected % 2 == 1).sum()),
                "untreated": n - len(selected),
                "true_profit": policy_totals(tau, value, cost, selected)["profit"],
                **solve,
            }
        )
    return rows


def _discount():
    cases = [
        ("high baseline", 0.70, 0.80, 40, 8, 1),
        ("low baseline", 0.10, 0.20, 40, 8, 1),
    ]
    return [
        {
            "case": name,
            "mu0": a,
            "mu1": b,
            "contribution": v,
            "discount": d,
            "delivery": c,
            "profit_without_discount": float(discount_profit(a, b, v, 0, c)),
            "profit_with_discount": float(discount_profit(a, b, v, d, c)),
            "expected_incentive_cost": b * d,
        }
        for name, a, b, v, d, c in cases
    ]


def _plot(target, rows):
    d = pd.DataFrame(rows)
    d = d[d.scenario.eq("E") & d.economy.isin(["uniform", "variable_cost"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for ax, (name, demo) in zip(axes, d.groupby("economy"), strict=True):
        for policy, p in demo.groupby("policy"):
            if policy in (
                "random",
                "predictive",
                "uplift",
                "profit",
                "integer",
                "oracle",
            ):
                s = p.sort_values("budget")
                ax.plot(s.budget, s.true_profit, marker="o", label=policy)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(
            title=f"Synthetic E · {name.replace('_', ' ')}",
            xlabel="Budget ceiling (sim. units)",
            ylabel="True expected profit (sim. units)",
        )
        ax.legend(fontsize=8)
    fig.savefig(target / "budget_frontier.png", dpi=145)
    plt.close(fig)


def run(root, *, fixture=False, output=None, seed=42):
    start = perf_counter()
    target = output or root / "artifacts" / (
        "session_11_fixture" if fixture else "session_11"
    )
    target.mkdir(parents=True, exist_ok=True)
    reliability = (
        {}
        if fixture
        else json.loads(
            (root / "artifacts/session_10/causal_reliability_report.json").read_text(
                encoding="utf-8"
            )
        )["datasets"]
    )
    st = (
        {}
        if fixture
        else _manifest(root / "artifacts/session_05/causal_model_manifest.json")
    )
    xr = (
        {}
        if fixture
        else _manifest(root / "artifacts/session_06/causal_model_manifest.json")
    )
    prior = (
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
    rows, solvers, distributions, assignments = [], [], [], []
    provenance = {}
    for scenario in "ABDEF":
        spec, train, val, truth = _frames(root, scenario, fixture, seed)
        tf = training_fingerprint(train, spec, "treatment")
        vf = training_fingerprint(val, spec, "treatment")
        if (
            not train.split.eq("train").all()
            or not val.split.eq("validation").all()
            or not val.unit_id.is_unique
            or set(train.unit_id) & set(val.unit_id)
        ):
            raise ValueError("Unexpected or overlapping model partitions")
        rel = reliability.get(scenario, {})
        if rel and (
            rel.get("train_fingerprint") != tf
            or rel.get("validation_fingerprint") != vf
        ):
            raise ValueError("Reliability and benchmark populations differ")
        dataset = f"synthetic_{scenario}"
        x = spec.features(val)
        constant = analyze_binary(train.treatment, train.outcome).absolute_lift
        labels = {
            "effect_model": "S-LightGBM",
            "comparison_model": "X-linear",
            "predictive_model": "pooled LightGBM",
        }
        if scenario == "F":
            current = (
                constant
                if fixture
                else pd.read_csv(
                    root / "artifacts/session_09/observational_ate_results.csv"
                )
                .loc[
                    lambda d: (
                        d.dataset.eq("F")
                        & d.method.eq("aipw")
                        & d.population.eq("validation")
                    ),
                    "estimate",
                ]
                .item()
            )
            effect = np.full(len(val), current)
            prediction = other = None
            labels = {
                "effect_model": "unsupported Scenario F aggregate AIPW score"
                if not fixture
                else "fixture arm contrast",
                "comparison_model": None,
                "predictive_model": None,
            }
        elif fixture:
            # Hypothetical feature-based scores exercise the engine without saved models.
            effect = 0.08 * np.tanh(x.engagement.to_numpy())
            prediction = (np.tanh(x.loyalty.to_numpy()) + 1) / 2
            other = None
            labels = {
                "effect_model": "fixture hypothetical score",
                "comparison_model": None,
                "predictive_model": "fixture hypothetical score",
            }
        else:
            fingerprint = _fingerprint(train, spec, "outcome")
            effect = _load_previous(
                root,
                st,
                (dataset, "outcome", "s_lightgbm"),
                fingerprint,
                spec.feature_columns,
                "05",
            )[0].predict_effect(x)
            other = _load_previous(
                root,
                xr,
                (dataset, "outcome", "x_linear"),
                fingerprint,
                spec.feature_columns,
                "06",
            )[0].predict_effect(x)
            prediction = _predictive_baseline(
                root, False, dataset, spec, train, val, "outcome", seed, prior
            )
        meta = statuses(reliability, scenario)
        provenance[scenario] = {
            "split": "validation",
            "units": len(val),
            "train_fingerprint": tf,
            "validation_fingerprint": vf,
            "feature_allowlist": list(spec.feature_columns),
            **labels,
            **meta,
        }
        for j, name in enumerate(SETTINGS if scenario != "F" else ("uniform",)):
            variable = name == "variable_cost"
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, ord(scenario), j])
            )
            n = min(180, len(val)) if variable else len(val)
            idx = (
                np.sort(rng.choice(len(val), n, replace=False))
                if n < len(val)
                else np.arange(n)
            )
            value, cost, eligible = _design(name, n, rng)
            a = effect[idx]
            unit = val.unit_id.to_numpy()[idx]
            econ = economics(unit, a, value, cost, eligible)
            capacity = int(0.30 * n)
            budgets = [int(n * 2 * f) for f in FRACTIONS]
            distributions.append(
                {
                    "scenario": scenario,
                    "economy": name,
                    "n": n,
                    "eligible_n": int(eligible.sum()),
                    "value_mean": float(value.mean()),
                    "value_min": int(value.min()),
                    "value_max": int(value.max()),
                    "cost_mean": float(cost.mean()),
                    "cost_min": int(cost.min()),
                    "cost_max": int(cost.max()),
                    "mean_break_even": float(econ.break_even.mean()),
                    "capacity": capacity,
                }
            )
            # Build learned and non-oracle policies before reading synthetic truth.
            selections = [
                _selections(
                    a,
                    value,
                    cost,
                    eligible,
                    b,
                    capacity,
                    None if prediction is None else prediction[idx],
                    constant,
                    seed,
                    variable,
                    None if other is None else other[idx],
                )
                for b in budgets
            ]
            if fixture:
                tau = truth.set_index("unit_id").loc[unit, "tau_true"].to_numpy()
            elif scenario == "F":
                indexed = pd.read_parquet(
                    truth, columns=["unit_id", "tau_true"]
                ).set_index("unit_id")
                if not indexed.index.is_unique:
                    raise ValueError("Duplicate truth unit identifiers")
                tau = indexed.loc[unit, "tau_true"].to_numpy()
            else:
                tau = aligned_synthetic_effects(val, spec, truth)[idx]
            for budget, learned in zip(budgets, selections, strict=True):
                oracle = (
                    None
                    if scenario == "F"
                    else allocate(
                        tau * value - cost,
                        cost,
                        eligible,
                        budget,
                        capacity=capacity,
                        policy="milp" if variable else "uniform",
                        seed=seed,
                    )
                )
                oracle_profit = (
                    policy_totals(tau, value, cost, oracle[0])["profit"]
                    if oracle is not None and oracle[1]["proven_optimal"]
                    else None
                )
                policies = learned if oracle is None else learned | {"oracle": oracle}
                for policy, (selected, solve) in policies.items():
                    pred = policy_totals(a, value, cost, selected)
                    true = policy_totals(tau, value, cost, selected)
                    key = {
                        "scenario": scenario,
                        "economy": name,
                        "budget": budget,
                        "capacity": capacity,
                        "policy": policy,
                        "population_n": n,
                        "scope": "bounded comparison"
                        if variable
                        else "full validation",
                    }
                    rows.append(
                        key
                        | meta
                        | {
                            "selected_count": len(selected),
                            "spend": solve["spend"],
                            **{f"predicted_{k}": v for k, v in pred.items()},
                            **{f"true_{k}": v for k, v in true.items()},
                            "oracle_regret": None
                            if oracle_profit is None
                            else max(0.0, oracle_profit - true["profit"]),
                        }
                    )
                    solvers.append(
                        key
                        | solve
                        | {
                            "objective_source": "true oracle profit"
                            if policy == "oracle"
                            else "constant effect profit"
                            if policy == "constant_effect"
                            else "common estimated profit"
                        }
                    )
                    assignments.extend(
                        key
                        | {
                            "unit_id": int(unit[i]),
                            "value": int(value[i]),
                            "cost": int(cost[i]),
                        }
                        for i in selected
                    )
    for name, table in (
        ("policy_optimization_results", rows),
        ("optimization_solver_results", solvers),
        ("customer_economics_summary", distributions),
        ("budget_frontier", rows),
        (
            "variable_cost_comparison",
            [r for r in rows if r["economy"] == "variable_cost"],
        ),
        ("discount_cost_demonstration", _discount()),
        ("multi_treatment_results", _multi(seed)),
    ):
        pd.DataFrame(table).to_csv(target / f"{name}.csv", index=False)
    pd.DataFrame(assignments).to_parquet(
        target / "local_assignments.parquet", index=False
    )
    write_json(
        target / "economic_scenarios.json",
        {
            "scenarios": SETTINGS,
            "seed": seed,
            "units": "simulated contribution units",
            "eligibility": "independent 90% probability",
            "budget": "floor(2 × population × fraction)",
            "budget_fractions": list(FRACTIONS),
            "capacity": "floor(0.30 × population)",
            "variable_cost_cap": 180,
        },
    )
    write_json(target / "policy_reliability_summary.json", provenance)
    _plot(target, rows)
    summary = {
        "runtime_seconds": perf_counter() - start,
        "seed": seed,
        "fixture": fixture,
        "policies": len(rows),
        "local_assignments": "local_assignments.parquet",
        "test_used": False,
    }
    write_json(target / "session_11_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    print(run(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
