import argparse
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from causallift.data.analytics import parquet_missing, parquet_summary, scenario_summary
from causallift.data.contract import DatasetSpec
from causallift.data.criteo import RAW_NAME, download, ingest
from causallift.data.splits import PARTITIONS, split_data
from causallift.simulation import SCENARIOS, simulate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def persist(frame: pd.DataFrame, spec: DatasetSpec, path: Path, seed: int):
    frame = frame.assign(split=split_data(frame, spec, seed))
    frame.to_parquet(path, index=False)
    write_json(path.with_suffix(".spec.json"), asdict(spec))

    arms = parquet_summary(path, spec)
    splits = parquet_summary(path, spec, by_split=True)
    splits.insert(0, "dataset", spec.dataset_id)
    splits["arm_proportion"] = splits["rows"] / splits.groupby("split")["rows"].transform("sum")
    missing = parquet_missing(path, tuple(frame.columns))
    quality = spec.quality(frame)
    if missing != quality["missing"]:
        raise RuntimeError(f"DuckDB missing counts disagree for {spec.dataset_id}")
    for keys, sql in (([spec.treatment_column], arms), (["split", spec.treatment_column], splits)):
        rate_names = (spec.outcome_column, *spec.secondary_outcomes)
        groups = frame.groupby(keys, sort=True)
        expected = groups.size().rename("rows").to_frame()
        for name in rate_names:
            expected[f"{name}_rate"] = groups[name].mean()
        columns = ["rows", *(f"{name}_rate" for name in rate_names)]
        np.testing.assert_allclose(sql[columns].to_numpy(), expected.to_numpy(), atol=1e-12)

    summary = {
        "spec": asdict(spec), "parquet": path.name, "rows": len(frame),
        "schema": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "split_sizes": {name: int(frame["split"].eq(name).sum()) for name in PARTITIONS},
        "arms": arms.assign(proportion=arms["rows"] / len(frame)).to_dict(orient="records"),
    }
    return summary, quality, splits


def plots(target: Path, examples: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, layout="constrained")
    for ax, scenario in zip(axes, ("A", "B"), strict=True):
        _, truth = examples[scenario]
        ax.hist(truth["tau_true"], bins=np.linspace(-0.2, 0.2, 41), color="#306C9C")
        ax.set(title=f"{scenario}. {SCENARIOS[scenario]}", xlabel="True CATE (probability difference)")
    axes[0].set_ylabel("Units")
    fig.savefig(target / "synthetic_effect_distribution.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, layout="constrained")
    for ax, scenario in zip(axes, ("A", "C", "F"), strict=True):
        observed, truth = examples[scenario]
        for arm, label, color, style in (
            (0, "Control", "#306C9C", "-"), (1, "Treated", "#B05227", "--"),
        ):
            values = truth.loc[observed["treatment"].eq(arm), "propensity_true"]
            weights = np.full(len(values), 1 / len(values))
            ax.hist(values, bins=np.linspace(0, 1, 41), weights=weights, histtype="step",
                    linewidth=1.8, linestyle=style, label=label, color=color)
        ax.set(title=f"{scenario}. {SCENARIOS[scenario]}", xlabel="True assignment probability")
    axes[0].set_ylabel("Fraction within arm per bin")
    axes[0].legend()
    fig.savefig(target / "propensity_overlap.png", dpi=150)
    plt.close(fig)


def run(root: Path, *, dataset: Path | None = None, fixture: bool = False,
        rows: int = 20_000, size: int = 200_000, seed: int = 42) -> None:
    target = root / "artifacts/session_01"
    synthetic = root / "data/synthetic"
    processed = root / "data/processed"
    for directory in (target, synthetic, processed, root / "data/raw"):
        directory.mkdir(parents=True, exist_ok=True)

    summaries, quality, splits, scenarios, examples = {}, {}, [], [], {}
    if dataset is not None:
        frame, spec, profile, source_quality = ingest(dataset, size, seed, fixture=fixture)
        name = spec.dataset_id
        summary, checks, split = persist(frame, spec, processed / f"{name}.parquet", seed)
        summaries[name] = summary | profile
        quality[name] = {"source": source_quality, "processed": checks}
        splits.append(split)

    for scenario in SCENARIOS:
        observed, truth, spec = simulate(scenario, rows, seed)
        name = spec.dataset_id
        path = synthetic / f"{name}.parquet"
        truth_path = synthetic / f"{name}_truth.parquet"
        summary, checks, split = persist(observed, spec, path, seed)
        truth.to_parquet(truth_path, index=False)
        summaries[name] = summary | {
            "ground_truth_file": truth_path.name,
            "ground_truth_schema": {column: str(dtype) for column, dtype in truth.dtypes.items()},
        }
        quality[name] = {
            "processed": checks,
            "ground_truth_missing": truth.isna().sum().astype(int).to_dict(),
        }
        splits.append(split)
        stats = scenario_summary(path, truth_path)
        treated = observed["treatment"].eq(1)
        expected = {
            "rows": len(observed), "treatment_proportion": treated.mean(),
            "outcome_prevalence": observed["outcome"].mean(),
            "true_ate": truth["tau_true"].mean(), "true_att": truth.loc[treated, "tau_true"].mean(),
            "naive_difference": observed.loc[treated, "outcome"].mean() - observed.loc[~treated, "outcome"].mean(),
            "cate_min": truth["tau_true"].min(), "cate_max": truth["tau_true"].max(),
        }
        np.testing.assert_allclose(list(stats.values()), [expected[key] for key in stats], atol=1e-12)
        scenarios.append({"scenario": scenario, "description": SCENARIOS[scenario], **stats})
        if scenario in ("A", "B", "C", "F"):
            examples[scenario] = (observed, truth)

    write_json(target / "dataset_summary.json", {
        "seed": seed,
        "python": platform.python_version(),
        "dependencies": {name: version(name) for name in (
            "numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "duckdb", "matplotlib",
        )},
        "datasets": summaries,
    })
    write_json(target / "data_quality_report.json", quality)
    pd.DataFrame(scenarios).to_csv(target / "synthetic_scenario_summary.csv", index=False)
    pd.concat(splits, ignore_index=True).to_csv(target / "split_summary.csv", index=False)
    plots(target, examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CausalLift Session 01 datasets and reports")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="Download or verify the publisher archive in data/raw")
    build = subparsers.add_parser("session01", help="Generate, validate, split and report datasets")
    source = build.add_mutually_exclusive_group()
    source.add_argument("--dataset", type=Path)
    source.add_argument("--synthetic-only", action="store_true")
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--rows", type=int, default=20_000, help="Rows per synthetic scenario")
    build.add_argument("--size", type=int, default=200_000, help="Criteo working subset size")
    args = parser.parse_args()
    if args.command == "download":
        print(download(args.root / "data/raw"))
        return
    default = args.root / "data/raw" / RAW_NAME
    dataset = None if args.synthetic_only else args.dataset or (default if default.is_file() else None)
    if dataset is None:
        print("Generating synthetic datasets. No Criteo input selected.")
    run(args.root, dataset=dataset, rows=args.rows, size=args.size, seed=args.seed)
    print(args.root / "artifacts/session_01")


if __name__ == "__main__":
    main()
