"""Train and report predictive baselines on fixed Session 01 partitions."""

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import matplotlib
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import CRITEO, FIXTURE
from causallift.data.splits import split_data
from causallift.prediction import (
    MODELS,
    calibration_bins,
    evaluate,
    fit_model,
    training_fingerprint,
)
from causallift.session01 import write_json
from causallift.simulation import simulate
from causallift.targeting import (
    describe_targeting,
    evaluate_synthetic_targeting,
    policy_rankings,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

COLORS = {"prevalence": "#707b86", "logistic": "#175c8c", "lightgbm": "#a14c27"}
POLICY_COLORS = {"random": "#6f7980", "predictive": "#a14c27", "constant_effect": "#7a5c91", "oracle": "#175c8c"}


def _prediction_plots(target: Path, curves: dict, calibration: list[dict]) -> None:
    kinds = {
        "precision_recall_curves.png": ("Recall", "Precision"),
        "roc_curves.png": ("False-positive rate", "True-positive rate"),
        "calibration_curves.png": ("Mean predicted probability", "Observed outcome rate"),
    }
    datasets = list(dict.fromkeys(dataset for dataset, _ in curves))
    outcomes = list(dict.fromkeys(outcome for _, outcome in curves))
    for filename, labels in kinds.items():
        fig, axes = plt.subplots(len(datasets), len(outcomes), squeeze=False,
                                 figsize=(5.5 * len(outcomes), 3.3 * len(datasets)), layout="constrained")
        for i, dataset in enumerate(datasets):
            for j, outcome in enumerate(outcomes):
                ax = axes[i, j]
                group = curves.get((dataset, outcome), {})
                if not group:
                    ax.set_visible(False)
                    continue
                if filename == "calibration_curves.png":
                    matching = [row for row in calibration if row["dataset"] == dataset and row["outcome"] == outcome]
                    limit = 1.06 * max(max(row["mean_probability"], row["observed_rate"]) for row in matching)
                    limit = limit or 0.01
                    ax.plot([0, limit], [0, limit], "--", color="#a5a5a5", label="Ideal")
                    for model in MODELS:
                        bins = [row for row in matching if row["model"] == model]
                        ax.plot([row["mean_probability"] for row in bins], [row["observed_rate"] for row in bins],
                                "o-", ms=3, color=COLORS[model], label=model)
                    ax.set(xlim=(0, limit), ylim=(0, limit))
                else:
                    for model, (y, p, metrics) in group.items():
                        if metrics["roc_auc"] is None:
                            continue
                        if filename == "roc_curves.png":
                            x, y_plot, _ = roc_curve(y, p)
                            value = metrics["roc_auc"]
                            ax.plot(x, y_plot, color=COLORS[model], label=f"{model} ({value:.4f})")
                        elif model == "prevalence":
                            ax.axhline(metrics["prevalence"], color=COLORS[model], linestyle="--",
                                      label=f"prevalence ({metrics['average_precision']:.4f})")
                        else:
                            y_plot, x, _ = precision_recall_curve(y, p)
                            value = metrics["average_precision"]
                            ax.step(x, y_plot, where="post", color=COLORS[model], label=f"{model} ({value:.4f})")
                    if filename == "precision_recall_curves.png" and dataset != "local_fixture" and outcome == "conversion":
                        # Rare-event curves need detail near baseline while retaining the full range.
                        ax.set_yscale("symlog", linthresh=0.001)
                ax.set(title=f"{dataset} · {outcome}", xlabel=labels[0], ylabel=labels[1])
                if ax.lines:
                    ax.legend(loc="best", fontsize=8)
        fig.savefig(target / filename, dpi=150)
        plt.close(fig)


def _targeting_plot(target: Path, rows: list[dict]) -> None:
    frame = pd.DataFrame(rows).dropna(subset=["mean_true_cate"])
    groups = list(frame.groupby("dataset", sort=False))
    fig, axes = plt.subplots(1, len(groups), squeeze=False, figsize=(3.3 * len(groups), 3.6), layout="constrained")
    for ax, (name, dataset) in zip(axes[0], groups, strict=True):
        for policy, group in dataset.groupby("policy", sort=False):
            style = "--" if policy == "constant_effect" else "-"
            ax.plot(100 * group["fraction"], group["expected_incremental_conversions"],
                    marker="o", ms=3, linestyle=style, color=POLICY_COLORS[policy], label=policy.replace("_", " "))
        ax.axhline(0, color="#9a9a9a", linewidth=0.7)
        ax.set(title=name.replace("synthetic", "Synthetic").replace("_", " "),
               xlabel="Eligible population targeted (%)", xticks=[10, 20, 50])
    axes[0, 0].set_ylabel("Expected incremental conversions")
    axes[0, -1].legend(fontsize=8, loc="best")
    fig.savefig(target / "targeting_comparison.png", dpi=150)
    plt.close(fig)


def _partitions(path: Path, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_dataset(path, spec, "train"), load_dataset(path, spec, "validation")


def _inputs(root: Path, temporary: Path, *, fixture: bool, development_only: bool, seed: int):
    if fixture:
        csv_path = root / "tests/fixtures/criteo_schema.csv"
        frame = pd.read_csv(csv_path)
        FIXTURE.validate(frame)
        frame["split"] = split_data(frame, FIXTURE, seed)
        yield FIXTURE.dataset_id, FIXTURE, frame.loc[frame["split"].eq("train")], frame.loc[
            frame["split"].eq("validation")], None, "tests/fixtures/criteo_schema.csv"
        observed, truth, spec = simulate("E", rows=600, seed=seed)
        observed["split"] = split_data(observed, spec, seed)
        truth_path = temporary / "fixture_truth.parquet"
        truth.to_parquet(truth_path, index=False)
        yield "synthetic_fixture_E", spec, observed.loc[observed["split"].eq("train")], observed.loc[
            observed["split"].eq("validation")], truth_path, "generated fixture E"
        return

    for label, relative in (
        ("Criteo 200k", "data/processed/criteo_v2_1.parquet"),
        ("Criteo 1M", "data/processed/benchmark_1m/data/processed/criteo_v2_1.parquet"),
    ):
        path = root / relative
        if label == "Criteo 1M" and (development_only or not path.is_file()):
            continue
        yield label, CRITEO, *_partitions(path, CRITEO), None, relative
    for scenario in "ABDE":
        relative = f"data/synthetic/synthetic_{scenario}.parquet"
        path = root / relative
        spec = DatasetSpec(**json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8")))
        yield spec.dataset_id, spec, *_partitions(path, spec), path.with_stem(f"{path.stem}_truth"), relative


def run(root: Path, *, output: Path | None = None, fixture: bool = False,
        development_only: bool = False, seed: int = 42) -> dict:
    target = output or root / "artifacts" / ("session_04_fixture" if fixture else "session_04")
    models_dir = target / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    versions = {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "threadpoolctl", "pyarrow", "matplotlib",
    )}
    real, synthetic, calibration, targeting, manifest = [], [], [], [], []
    curves = {}
    with TemporaryDirectory() as scratch:
        for dataset, spec, train, validation, truth_path, source in _inputs(
            root, Path(scratch), fixture=fixture, development_only=development_only, seed=seed,
        ):
            predictions = {}
            results = synthetic if truth_path is not None else real
            for outcome in (spec.outcome_column, *spec.secondary_outcomes):
                y = validation[outcome].to_numpy()
                fingerprint = training_fingerprint(train, spec, outcome)
                for name in MODELS:
                    start = perf_counter()
                    model = fit_model(name, spec, train, validation, outcome, seed=seed,
                                      estimators=35 if fixture else 160)
                    duration = perf_counter() - start
                    p = model.predict(validation)
                    predictions[outcome, name] = p
                    metrics = evaluate(y, p)
                    stem = dataset.lower().replace(" ", "_")
                    model_path = models_dir / f"{stem}_{outcome}_{name}.joblib"
                    joblib.dump(model, model_path, compress=3)
                    identity = {"dataset": dataset, "outcome": outcome, "model": name}
                    info = identity | {"train_n": len(train), "train_positives": int(train[outcome].sum()),
                                       "train_prevalence": float(train[outcome].mean()), "fit_seconds": duration,
                                       "best_iteration": model.best_iteration}
                    results.append(info | metrics)
                    manifest.append(info | {
                        "model_path": model_path.relative_to(target).as_posix(), "configuration": model.configuration,
                        "training_fingerprint_sha256": fingerprint, "feature_allowlist": list(spec.feature_columns),
                        "split_provenance": {"input": source, "training": "train", "evaluation": "validation",
                                             "early_stopping": "validation" if name == "lightgbm" else None},
                        "seed": seed, "python": platform.python_version(), "dependencies": versions, "metrics": metrics,
                    })
                    calibration.extend(identity | row for row in calibration_bins(y, p))
                    if truth_path is None:
                        curves.setdefault((dataset, outcome), {})[name] = (y, p, metrics)
            probability = predictions[spec.outcome_column, "lightgbm"]
            rankings, effect = policy_rankings(probability, train, spec, seed=seed)
            if truth_path is None:
                rows = describe_targeting(rankings, probability, validation, spec)
            else:
                rows = evaluate_synthetic_targeting(rankings, probability, validation, spec, truth_path, seed=seed)
            targeting.extend({"dataset": dataset, "predictive_model": "lightgbm",
                              "training_constant_effect_score": effect, **row} for row in rows)

    for filename, rows in (
        ("predictive_model_results.csv", real), ("synthetic_predictive_results.csv", synthetic),
        ("calibration_summary.csv", calibration), ("targeting_baseline_results.csv", targeting),
    ):
        pd.DataFrame(rows).to_csv(target / filename, index=False)
    summary = {
        "seed": seed, "fixture": fixture, "python": platform.python_version(), "dependencies": versions,
        "models": len(manifest), "evaluated_partitions": ["validation"],
        "early_stopping_note": "LightGBM uses validation log loss for early stopping. Metrics are validation benchmarks.",
        "prediction_estimand": "Pooled observed outcome probability under the logged assignment mixture",
        "targeting_note": "Policy capacities match within each dataset and fraction. Synthetic CATE is evaluation-only.",
        "targeting_scoring_model": "lightgbm",
        "real_results": real, "synthetic_results": synthetic, "targeting_results": targeting,
    }
    write_json(target / "predictive_model_manifest.json", {"models": manifest})
    write_json(target / "session_04_summary.json", summary)
    _prediction_plots(target, curves, calibration)
    _targeting_plot(target, targeting)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--development-only", action="store_true", help="Skip the optional Criteo 1M input")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))
    print(args.output or args.root / "artifacts" / ("session_04_fixture" if args.fixture else "session_04"))


if __name__ == "__main__":
    main()
