"""Generate experiment-design and health reports from existing benchmark inputs."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import CRITEO
from causallift.design_simulation import (
    MULTIPLICITY_METHODS,
    cuped_examples,
    run_simulations,
)
from causallift.experiment_decisions import ExperimentSpec, Guardrail, decide_experiment
from causallift.experiment_health import (
    adjust_p_values,
    covariate_balance,
    sample_ratio_mismatch,
)
from causallift.experimentation import binary_from_counts
from causallift.planning import (
    binary_mde,
    binary_sample_size,
    continuous_sample_size,
    validate_counts,
)
from causallift.session01 import write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _planning(archive: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    plans, mdes = [], []
    for row in archive.itertuples():
        baseline = row.control_events / row.control_n
        for share in (0.5, 0.85):
            for power in (0.8, 0.9):
                inputs = {"outcome": row.outcome, "baseline": baseline, "target_treatment_share": share,
                          "alpha": 0.05, "target_power": power}
                for relative in (0.05, 0.10, 0.20):
                    alternative = baseline * (1 + relative)
                    plan = binary_sample_size(baseline, alternative, ratio=share / (1 - share), target_power=power)
                    plans.append(inputs | {"alternative": alternative, "absolute_effect": alternative - baseline,
                                           "relative_effect": relative, **asdict(plan), "total_n": plan.total_n})
                for total in (200_000, 1_000_000):
                    nt = round(total * share)
                    mde = binary_mde(baseline, total - nt, nt, target_power=power)
                    mdes.append(inputs | {"total_n": total, "control_n": total - nt, "treated_n": nt, **asdict(mde)})
    return plans, mdes


def _balance(root: Path) -> list[dict]:
    datasets = [
        (root / "data/processed/criteo_v2_1.parquet", CRITEO, ("f0", "f1", "f2")),
    ]
    for scenario in ("A", "C"):
        path = root / f"data/synthetic/synthetic_{scenario}.parquet"
        spec = DatasetSpec(**json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8")))
        datasets.append((path, spec, ("engagement", "loyalty", "spend")))
    return [
        row | {"partition": "train"}
        for path, spec, features in datasets
        for row in covariate_balance(load_dataset(path, spec, "train"), spec, features)
    ]


def _multiple_testing(groups: pd.DataFrame) -> tuple[list[dict], dict]:
    if groups.empty or not groups["partition"].eq("validation").all():
        raise ValueError("Expected the Session 02 validation-only subgroup report")
    results, counts = [], {}
    # Each dataset and feature defines its own family of prespecified bins.
    for (dataset, feature), family in groups.groupby(["dataset", "feature"], sort=False):
        identifier = f"{dataset}/{feature}"
        # A missing test stays in the family with p=1 for conservative adjustment.
        raw = family["p_value"].fillna(1).to_numpy()
        counts[identifier] = {"uncorrected": int((raw <= 0.05).sum())}
        for method in MULTIPLICITY_METHODS:
            corrected = adjust_p_values(raw, method)
            counts[identifier][method] = sum(corrected.reject)
            for row, adjusted, reject in zip(family.itertuples(), corrected.adjusted_p_values,
                                             corrected.reject, strict=True):
                missing = pd.isna(row.p_value)
                results.append({
                    "family": identifier, "feature": feature, "dataset": dataset, "subgroup": row.subgroup,
                    "partition": "validation", "exploratory": True, "family_size": corrected.family_size,
                    "raw_p": None if missing else row.p_value, "adjusted_p": None if missing else adjusted,
                    "raw_reject": None if missing else row.p_value <= corrected.alpha,
                    "adjusted_reject": None if missing else reject, "method": method,
                    "error_rate": corrected.error_rate, "alpha": corrected.alpha,
                })
    return results, counts


def _guardrail_examples() -> dict:
    spec = ExperimentSpec("conversion", "higher", minimum_practical_effect=0.005,
                          guardrails=(Guardrail("refund", "lower", 0.005),))
    primary = binary_from_counts(1200, 10000, 1000, 10000)
    cases = {"primary_and_guardrail_pass": (200, "pass"), "guardrail_harm": (350, "pass"),
             "guardrail_inconclusive": (220, "pass"), "experiment_health_failed": (200, "fail")}
    output = {}
    for name, (events, health) in cases.items():
        refund = binary_from_counts(events, 10000, 200, 10000)
        decision = decide_experiment(spec, {"conversion": primary, "refund": refund}, health=health)
        output[name] = decision | {"synthetic": True, "primary_estimate": primary.absolute_lift,
                                  "guardrail_estimate": refund.absolute_lift}
    return output


def _plots(target: Path, balance: list[dict], simulations: list[dict]) -> None:
    frame = pd.DataFrame(balance)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True, layout="constrained")
    for ax, (dataset, rows) in zip(axes, frame.groupby("dataset", sort=False), strict=True):
        for sign in (-1, 1):
            ax.axvline(sign * rows["threshold"].iloc[0], color="#888888", linestyle="--", linewidth=1)
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.scatter(rows["smd"], rows["feature"], color="#25628f")
        ax.set(title={"criteo_v2_1": "Criteo", "synthetic_A": "Synthetic A · randomized",
                      "synthetic_C": "Synthetic C · confounded"}[dataset], xlabel="Treated − control SMD")
        ax.invert_yaxis()
    fig.suptitle("Pre-treatment balance on training rows")
    fig.savefig(target / "covariate_balance.png", dpi=155)
    plt.close(fig)

    labels = ["Fixed horizon", "Naive peeking", "Uncorrected", "Bonferroni", "Holm", "Benjamini–Hochberg"]
    keys = ("fixed_horizon", "naive_peeking", "uncorrected", *MULTIPLICITY_METHODS)
    by_name = {row["check"]: row for row in simulations}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, subset, title in zip(axes, (slice(0, 2), slice(2, 6)),
                                 ("Repeated looks under one null", "Families of 20 true nulls"), strict=True):
        rows = [by_name[name] for name in keys[subset]]
        values = [100 * row["estimate"] for row in rows]
        errors = [196 * row["monte_carlo_se"] for row in rows]
        ax.barh(labels[subset], values, xerr=errors, capsize=3, color="#25628f")
        ax.axvline(5, color="#9c4b26", linestyle="--", linewidth=1)
        ax.set(title=title, xlabel="Experiments or families with a rejection (%)")
        ax.invert_yaxis()
    fig.savefig(target / "statistical_simulation_summary.png", dpi=155)
    plt.close(fig)


def run(root: Path, *, output: Path | None = None, seed: int = 42, repetitions: int = 3000) -> dict:
    validate_counts(repetitions, minimum=2)
    previous = root / "artifacts/session_02"
    criteo = pd.read_csv(previous / "criteo_experiment_results.csv")
    archive = criteo.loc[criteo["dataset"].eq("Full archive")]
    if sorted(archive["outcome"]) != ["conversion", "visit"]:
        raise ValueError("Expected one full-archive result per Criteo outcome")
    plans, mdes = _planning(archive)
    balance = _balance(root)
    corrections, rejection_counts = _multiple_testing(pd.read_csv(previous / "subgroup_analysis.csv"))
    conversion = archive.loc[archive["outcome"].eq("conversion")].iloc[0]
    srm = {
        "healthy_synthetic": sample_ratio_mismatch(510, 490, 0.5),
        "mismatched_synthetic": sample_ratio_mismatch(650, 350, 0.5),
        "unequal_synthetic": sample_ratio_mismatch(850, 150, 0.15),
        "criteo_released_archive": sample_ratio_mismatch(int(conversion.control_n), int(conversion.treated_n),
                                                          CRITEO.assignment_probability),
    }
    guardrails = _guardrail_examples()
    # Separate streams for the binary tests and CUPED demonstrations.
    simulation_seed, cuped_seed = (int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(2))
    simulation = run_simulations(simulation_seed, repetitions)
    cuped, cuped_trials = cuped_examples(cuped_seed, repetitions)
    simulations = simulation + [row | {"seed": cuped_seed, "repetitions": repetitions} for row in cuped_trials]
    continuous = continuous_sample_size(0.2, target_power=0.8)
    summary = {
        "seed": seed, "repetitions": repetitions,
        "planning_scope": "Hypothetical future experiments using exact Session 02 control rates.",
        "binary_planning_method": "two_sided_normal_cohen_h",
        "planning_rows": len(plans), "mde_rows": len(mdes),
        "continuous_example": {"standardized_effect": 0.2, "alpha": 0.05, "target_power": 0.8,
                               "assumption": "Common population variance", **asdict(continuous),
                               "total_n": continuous.total_n},
        "srm_statuses": {name: result.status for name, result in srm.items()},
        "balance_flags": {name: int(rows["flag"].eq(True).sum())
                          for name, rows in pd.DataFrame(balance).groupby("dataset")},
        "subgroup_rejections": rejection_counts,
        "guardrail_decisions": {name: result["decision"] for name, result in guardrails.items()},
        "global_null_note": "FDR equals FWER under the global null because every rejection is false.",
        "simulations": simulations,
    }
    target = output or root / "artifacts/session_03"
    target.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("experiment_planning.csv", plans), ("mde_analysis.csv", mdes),
        ("covariate_balance.csv", balance), ("multiple_testing_results.csv", corrections),
        ("monte_carlo_results.csv", [row | {"parameters": json.dumps(row["parameters"])} for row in simulations]),
    ):
        pd.DataFrame(rows).to_csv(target / filename, index=False)
    write_json(target / "srm_results.json", {name: asdict(result) for name, result in srm.items()})
    write_json(target / "guardrail_decisions.json", guardrails)
    write_json(target / "cuped_results.json", {name: asdict(result) for name, result in cuped.items()})
    write_json(target / "experiment_design_summary.json", summary)
    _plots(target, balance, simulations)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=3000)
    args = parser.parse_args()
    run(**vars(args))
    print(args.output or args.root / "artifacts/session_03")


if __name__ == "__main__":
    main()
