"""Run classical experiment reports on the materialized Session 01 datasets."""

import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from causallift.data.analytics import binary_arm_counts
from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import CRITEO, FIXTURE, RAW_NAME, SHA256, checksum
from causallift.experimentation import (
    ExperimentResult,
    analyze_binary,
    analyze_continuous,
    binary_from_counts,
)
from causallift.simulation import SCENARIOS
from causallift.subgroups import subgroup_analysis

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

CRITEO_INTERPRETATION = "Released-benchmark assignment contrast under documented identification assumptions"
RANDOMIZED_INTERPRETATION = "Randomized assignment effect"
OBSERVATIONAL_INTERPRETATION = "Observational association; naive contrast does not identify ATE"
CRITEO_LIMITATIONS = (
    "Privacy subsampling limits interpretation to the released benchmark.",
    "Pooled trials lack trial and customer identifiers, so clustering cannot be checked.",
    "Working samples are nested in the same archive and are not independent replications.",
    "Standard errors use an independent-row approximation and omit benchmark construction uncertainty.",
)


def _report_fields(result: ExperimentResult) -> dict:
    low, high = result.confidence_interval
    wald_low, wald_high = result.wald_interval
    bootstrap_low, bootstrap_high = result.bootstrap_interval or (None, None)
    return {
        "dataset": result.dataset_id, "outcome": result.experiment_id, "total_n": result.total_n,
        "treated_n": result.treatment.n, "control_n": result.control.n,
        "treated_events": result.treatment.events, "control_events": result.control.events,
        "treated_rate": result.treatment.mean, "control_rate": result.control.mean,
        "absolute_lift": result.absolute_lift, "relative_lift": result.relative_lift,
        "effect_se": result.effect_se, "null_se": result.null_se,
        "statistic": result.statistic, "statistic_name": result.statistic_name,
        "p_value": result.p_value, "log10_p_value": result.log10_p_value,
        "confidence_level": result.confidence_level, "ci_low": low, "ci_high": high,
        "wald_low": wald_low, "wald_high": wald_high,
        "bootstrap_low": bootstrap_low, "bootstrap_high": bootstrap_high,
        "significant": result.significant, "test_method": result.test_method,
        "interval_method": result.interval_method, "warnings": " | ".join(result.warnings),
        "assumptions": " | ".join(result.assumptions),
    }


def _arm_report(path: Path, spec: DatasetSpec, dataset: str, options: dict, *, csv: bool = False):
    outcomes = (spec.outcome_column, *spec.secondary_outcomes)
    control, treated = binary_arm_counts(path, spec, csv=csv)
    interpretation = "Fixture demonstration" if spec.source_type == "fixture" else CRITEO_INTERPRETATION
    return [
        _report_fields(binary_from_counts(
            tx, treated[0], cx, control[0], dataset_id=dataset, experiment_id=outcome, **options,
        )) | {"interpretation": interpretation}
        for outcome, tx, cx in zip(outcomes, treated[1:], control[1:], strict=True)
    ]


def _synthetic_report(root: Path, options: dict):
    reports = []
    subgroup = None
    for scenario, description in SCENARIOS.items():
        path = root / f"data/synthetic/synthetic_{scenario}.parquet"
        spec = DatasetSpec(**json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8")))
        observed = load_dataset(path, spec)
        truth = pd.read_parquet(path.with_stem(f"{path.stem}_truth"), columns=[spec.unit_id, "tau_true"])
        if len(observed) != len(truth) or truth[spec.unit_id].isna().any():
            raise ValueError(f"Observed rows and truth do not align for {spec.dataset_id}")
        joined = observed.merge(truth, on=spec.unit_id, how="left", validate="one_to_one")
        if not np.isfinite(joined["tau_true"]).all():
            raise ValueError(f"Missing or invalid effect truth for {spec.dataset_id}")
        result = analyze_binary(observed[spec.treatment_column], observed[spec.outcome_column],
                                dataset_id=spec.dataset_id, experiment_id=spec.outcome_column, **options)
        true_ate = float(joined["tau_true"].mean())
        randomized = spec.source_type == "synthetic_randomized"
        reports.append(_report_fields(result) | {
            "scenario": scenario, "description": description, "true_ate": true_ate,
            "true_att": float(joined.loc[joined[spec.treatment_column].eq(1), "tau_true"].mean()),
            "estimation_error": result.absolute_lift - true_ate, "identifies_ate": randomized,
            "interpretation": RANDOMIZED_INTERPRETATION if randomized else OBSERVATIONAL_INTERPRETATION,
        })
        if scenario == "E":
            subgroup = (observed, spec)
    return reports, subgroup


def _continuous_demo(options: dict) -> dict:
    rng = np.random.default_rng(options["seed"])
    # Fixed randomized arm sizes with independently generated, unequal-variance revenue.
    assignment = rng.permutation(np.repeat([0, 1], [400, 300]))
    revenue = rng.gamma(shape=np.where(assignment, 3, 4), scale=np.where(assignment, 6.0, 4.0))
    result = analyze_continuous(assignment, revenue, **options, dataset_id="synthetic_revenue",
                                experiment_id="revenue")
    return asdict(result) | {"total_n": result.total_n, "true_mean_difference": 2.0,
                            "interpretation": RANDOMIZED_INTERPRETATION}


def _draw_intervals(ax, rows: list[dict], labels: list[str], title: str) -> None:
    ax.axvline(0, color="#58636e", linestyle="--", linewidth=1)
    for index, row in enumerate(rows):
        color = "#9c4b26" if row.get("identifies_ate") is False else "#25628f"
        ax.hlines(index, 100 * row["ci_low"], 100 * row["ci_high"], color=color, linewidth=2)
        ax.plot(100 * row["absolute_lift"], index, "o", color=color)
    ax.set(yticks=range(len(labels)), yticklabels=labels, title=title, xlabel="Treated − control (pp)")
    ax.invert_yaxis()


def _plot(path: Path, criteo: list[dict], synthetic: list[dict], confidence: float) -> None:
    panels = []
    for outcome in ("conversion", "visit"):
        rows = [row for row in criteo if row["outcome"] == outcome]
        if rows:
            dataset = "Fixture" if rows[0]["dataset"] == FIXTURE.dataset_id else "Criteo"
            panels.append((rows, [row["dataset"] for row in rows], f"{dataset} {outcome}"))
    if synthetic:
        panels.append((synthetic, [row["scenario"] + (" · association" if not row["identifies_ate"] else "")
                                   for row in synthetic], "Synthetic outcomes"))
    fig = plt.figure(figsize=(10, 6 if len(panels) == 3 else 3.5), layout="constrained")
    if len(panels) == 3:
        grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])
        axes = [fig.add_subplot(place) for place in (grid[0, 0], grid[0, 1], grid[1, :])]
    else:
        axes = list(fig.subplots(1, len(panels), squeeze=False)[0])
    for ax, panel in zip(axes, panels, strict=True):
        _draw_intervals(ax, *panel)
    fig.suptitle(f"Assignment contrasts with {confidence:.1%} Newcombe intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(
    root: Path, *, mode: str = "full", output: Path | None = None,
    fixture: Path | None = None, seed: int = 42, confidence_level: float = 0.95,
    bootstrap_resamples: int = 2000,
) -> dict:
    if mode not in {"full", "synthetic", "fixture"}:
        raise ValueError("Mode must be full, synthetic or fixture")
    options = {"seed": seed, "confidence_level": confidence_level, "bootstrap_resamples": bootstrap_resamples}
    continuous = _continuous_demo(options)
    criteo, synthetic, subgroups = [], [], []
    inputs = []
    if mode == "full":
        raw = root / "data/raw" / RAW_NAME
        if checksum(raw) != SHA256:
            raise ValueError("Criteo archive SHA-256 does not match the verified publisher release")
        working = root / "data/processed/criteo_v2_1.parquet"
        larger = root / "data/processed/benchmark_1m/data/processed/criteo_v2_1.parquet"
        for label, path, csv, expected in (
            ("Full archive", raw, True, 13_979_592),
            ("200k sample", working, False, 200_000),
            ("1M sample", larger, False, 1_000_000),
        ):
            if label == "1M sample" and not path.is_file():
                continue
            rows = _arm_report(path, CRITEO, label, options, csv=csv)
            if rows[0]["total_n"] != expected:
                raise ValueError(f"{label} row count differs from the Session 01 benchmark designation")
            criteo.extend(rows)
            inputs.append(str(path.relative_to(root)))
        train = load_dataset(working, CRITEO, "train")
        validation = load_dataset(working, CRITEO, "validation")
        threshold = float(train["f0"].median())
        subgroups.extend(row | {"partition": "validation", "boundary_origin": "training f0 median",
                                "boundary": threshold} for row in subgroup_analysis(
            validation, CRITEO, "f0", bins=(-np.inf, threshold, np.inf), confidence_level=confidence_level,
        ))
    if mode in ("full", "synthetic"):
        synthetic, subgroup = _synthetic_report(root, options)
        observed, spec = subgroup
        subgroups.extend(row | {"partition": "validation", "boundary_origin": "prespecified zero",
                                "boundary": 0.0} for row in subgroup_analysis(
            observed.loc[observed["split"].eq("validation")], spec, "price_sensitivity",
            bins=(-np.inf, 0.0, np.inf), confidence_level=confidence_level,
        ))
        inputs.append("data/synthetic/synthetic_{A..F}{,_truth}.parquet")
    if mode == "fixture":
        fixture = fixture or root / "tests/fixtures/criteo_schema.csv"
        criteo = _arm_report(fixture, FIXTURE, FIXTURE.dataset_id, options, csv=True)
        inputs.append(str(fixture))

    summary = {
        "mode": mode, **options, "inputs": inputs, "python": platform.python_version(),
        "dependencies": {name: version(name) for name in ("numpy", "pandas", "scipy", "duckdb", "matplotlib")},
        "criteo_archive_sha256": SHA256 if mode == "full" else None,
        "criteo_limitations": CRITEO_LIMITATIONS if mode == "full" else (),
        "subgroup_note": "Exploratory, unadjusted comparisons. Differences in significance cannot establish heterogeneity.",
        "benchmark_note": "Full-cohort summaries are descriptive benchmarks. No model tuning used the test split.",
        "criteo_comparisons": criteo, "synthetic_comparisons": synthetic, "subgroups": subgroups,
    }
    target = output or root / "artifacts" / ("session_02" if mode == "full" else f"session_02_{mode}")
    target.mkdir(parents=True, exist_ok=True)
    csv_names = (
        ("criteo_experiment_results.csv" if mode == "full" else "fixture_experiment_results.csv", criteo),
        ("synthetic_experiment_results.csv", synthetic), ("subgroup_analysis.csv", subgroups),
    )
    for name, rows in csv_names:
        if rows:
            pd.DataFrame(rows).to_csv(target / name, index=False)
    for name, payload in (("experiment_summary.json", summary), ("continuous_experiment_results.json", continuous)):
        (target / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _plot(target / "treatment_effect_intervals.png", criteo, synthetic, confidence_level)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--mode", choices=("full", "synthetic", "fixture"), default="full")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
