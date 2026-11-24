"""Benchmark partially linear DML, LinearDML and an honest causal forest."""

import argparse
import hashlib
import platform
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causallift.confounding_demo import confounding_study
from causallift.double_ml import AdvancedDML, partially_linear_dml
from causallift.experimentation import analyze_binary
from causallift.meta_learners import MetaLearner
from causallift.prediction import training_fingerprint
from causallift.residual_learners import ResidualLearner, training_propensity
from causallift.session01 import write_json
from causallift.session05 import METADATA, PAIRS, _benchmark, _predictive_baseline
from causallift.session06 import XR, _cohorts, _manifest
from causallift.targeting import aligned_synthetic_effects
from causallift.uplift import effect_accuracy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

ADVANCED = ("linear_dml", "causal_forest")


def _training_subset(train, spec, outcome, ceiling, seed):
    if len(train) <= ceiling:
        return train
    strata = 2 * train[spec.treatment_column] + train[outcome]
    take, _ = train_test_split(np.arange(len(train)), train_size=ceiling, random_state=seed, stratify=strata)
    return train.iloc[np.sort(take)]


def _fingerprint(train, spec, outcome):
    digest = hashlib.sha256(bytes.fromhex(training_fingerprint(train, spec, outcome)))
    digest.update(train[spec.treatment_column].to_numpy(dtype="int8").tobytes())
    return digest.hexdigest()


def _load_previous(root, entries, key, fingerprint, columns, session):
    record = entries[key]
    if (record["training_fingerprint_sha256"] != fingerprint
            or tuple(record["feature_allowlist"]) != tuple(columns)):
        raise ValueError(f"Saved model provenance differs for {key}")
    model = joblib.load(root / f"artifacts/session_{session}" / record["model_path"])
    return model, record["fit_seconds"]


def _interval_summary(model, xv, tau, seed):
    n = len(xv)
    pick = np.arange(n) if n <= 4000 or tau is not None else np.sort(
        np.random.default_rng(seed).choice(n, 4000, replace=False),
    )
    sample = xv.iloc[pick]
    interval_warnings = list(model.fit_warnings_)
    lo = hi = np.full(len(sample), np.nan)
    avg_lo = avg_hi = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if not any("underdermined" in note or "underdetermined" in note for note in interval_warnings):
            lo, hi = model.effect_interval(sample)
        if lo.shape != (len(sample),) or hi.shape != lo.shape:
            raise ValueError("Invalid library interval dimensions")
        supported = np.isfinite(lo) & np.isfinite(hi) & (lo <= hi)
        if supported.any():
            candidate = model.average_interval(sample.iloc[supported])
            if np.isfinite(candidate).all() and candidate[0] < candidate[1]:
                avg_lo, avg_hi = candidate
    interval_warnings.extend(str(item.message) for item in caught)
    if not supported.all():
        interval_warnings.append("Some pointwise intervals were unsupported. Coverage and average inference use supported rows.")
    if avg_lo is None:
        interval_warnings.append("The library did not provide a valid average-effect interval.")
    supported_n = int(supported.sum())
    return {
        "interval_population_n": len(pick), "validation_population_n": n,
        "supported_interval_n": supported_n,
        "interval_population": "full_validation" if len(pick) == n else "seeded_validation_subset",
        "mean_pointwise_width": float(np.mean(hi[supported] - lo[supported])) if supported_n else None,
        "confidence_level": 0.95,
        "pointwise_coverage_true_cate": None if tau is None or not supported_n else float(
            np.mean((lo[supported] <= tau[pick][supported]) & (tau[pick][supported] <= hi[supported])),
        ),
        "mean_effect_on_interval_population": float(model.predict_effect(sample).mean()),
        "mean_effect_on_supported_rows": float(model.predict_effect(sample.iloc[supported]).mean()) if supported_n else None,
        "average_effect_ci_low": avg_lo, "average_effect_ci_high": avg_hi,
        "average_inference": "EconML ate_interval on supported rows of the stated interval population",
        "warnings": " | ".join(dict.fromkeys(interval_warnings)),
    }


def _plots(path, accuracy, confounded):
    a = pd.DataFrame(accuracy)
    h = pd.DataFrame(confounded)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), layout="constrained")
    pehe = a.pivot(index="policy", columns="dataset", values="root_pehe") * 100
    im = axes[0].imshow(pehe, aspect="auto", cmap="Blues")
    axes[0].set(xticks=range(len(pehe.columns)), xticklabels=[c.rsplit("_", 1)[-1] for c in pehe.columns],
                yticks=range(len(pehe)), yticklabels=[p.replace("_", " ") for p in pehe.index],
                title="Synthetic root PEHE (pp)", xlabel="Scenario")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    for i, (method, group) in enumerate(h.groupby("method", sort=False)):
        axes[1].scatter(group["estimate"], np.full(len(group), i), alpha=0.45, s=22)
        axes[1].plot(group["estimate"].mean(), i, "|", ms=17, color="#222222")
    axes[1].axvline(h["true_effect"].iloc[0], color="#a34d27", linestyle="--", label="True effect")
    axes[1].set(yticks=range(h["method"].nunique()), yticklabels=[m.replace("_", " ") for m in h["method"].unique()],
                xlabel="Estimated effect", title="Controlled confounding")
    axes[1].legend(loc="best")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root: Path, *, output: Path | None = None, fixture: bool = False, development_only: bool = False,
        seed: int = 42, forest_max_train: int = 120_000, repetitions: int = 30) -> dict:
    try:
        from econml.dml import CausalForestDML  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Session 07 advanced benchmarks require the [advanced] extra") from error
    if forest_max_train < 300 or repetitions < 1:
        raise ValueError("Forest ceiling must be at least 300 and repetitions positive")
    start = perf_counter()
    target = output or root / "artifacts" / ("session_07_fixture" if fixture else "session_07")
    (target / "models").mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "dependencies": {name: version(name) for name in (
        "numpy", "scipy", "scikit-learn", "lightgbm", "pandas", "pyarrow", "joblib", "econml",
        "numba", "llvmlite", "shap", "sparse", "statsmodels", "slicer", "tqdm",
    )}}
    old_st = {} if fixture else _manifest(root / "artifacts/session_05/causal_model_manifest.json")
    old_xr = {} if fixture else _manifest(root / "artifacts/session_06/causal_model_manifest.json")
    prior = {} if fixture else {key[:2]: item for key, item in _manifest(
        root / "artifacts/session_04/predictive_model_manifest.json", "model", "lightgbm",
    ).items()}
    structural, accuracy, ranks, targets, calibration, uncertainty, manifest, comparisons = ([] for _ in range(8))
    diagnostics = {}
    with TemporaryDirectory() as temporary:
        for dataset, spec, train, val, truth_path, source in _cohorts(
            root, Path(temporary), fixture=fixture, development_only=development_only, seed=seed,
        ):
            propensity = training_propensity(spec, train)
            xt, xv = spec.features(train), spec.features(val)
            tt = train[spec.treatment_column]
            groups = train[spec.unit_id] if spec.unit_id else None
            for outcome in (spec.outcome_column, *spec.secondary_outcomes):
                identity = {"dataset": dataset, "outcome": outcome}
                yt = train[outcome]
                fingerprint = _fingerprint(train, spec, outcome)
                predicted = _predictive_baseline(root, fixture, dataset, spec, train, val, outcome, seed, prior)
                ate = analyze_binary(tt, yt).absolute_lift
                effects = {"constant_effect": np.full(len(val), ate)}
                comparison_details = {"constant_effect": {"train_n": len(train), "fit_seconds": 0.0,
                                                          "model_origin": "training arm means"}}
                for learner, base in (*PAIRS, *XR):
                    policy = f"{learner}_{base}"
                    is_st = learner in ("s", "t")
                    if fixture:
                        model = (MetaLearner(learner, base, seed=seed, estimators=16) if is_st else
                                 ResidualLearner(learner, base, propensity, seed=seed, estimators=16))
                        begin = perf_counter()
                        model.fit(xt, tt, yt)
                        elapsed = perf_counter() - begin
                    else:
                        model, elapsed = _load_previous(root, old_st if is_st else old_xr,
                            (dataset, outcome, policy), fingerprint, spec.feature_columns, "05" if is_st else "06")
                    effects[policy] = model.predict_effect(xv)
                    comparison_details[policy] = {"train_n": len(train), "fit_seconds": elapsed,
                        "model_origin": "fixture" if fixture else ("session_05" if is_st else "session_06")}
                begin = perf_counter()
                pl, detail = partially_linear_dml(xt, tt, yt, propensity=propensity.value, groups=groups, seed=seed)
                pl_seconds = perf_counter() - begin
                diagnostics[f"{dataset}/{outcome}"] = identity | detail
                advanced = {}
                for policy in ADVANCED:
                    fitting = _training_subset(train, spec, outcome, forest_max_train, seed) if policy == "causal_forest" else train
                    xfit = xt if fitting is train else spec.features(fitting)
                    begin = perf_counter()
                    estimator = AdvancedDML(policy, propensity.value, seed=seed, trees=24 if fixture else 100,
                                            min_leaf=8 if fixture else 35)
                    estimator.fit(xfit, fitting[spec.treatment_column], fitting[outcome])
                    elapsed = perf_counter() - begin
                    effect = estimator.predict_effect(xv)
                    effects[policy] = effect
                    relpath = Path("models") / f"{dataset.lower().replace(' ', '_')}_{outcome}_{policy}.joblib"
                    joblib.dump(estimator, target / relpath, compress=3)
                    np.testing.assert_allclose(joblib.load(target / relpath).predict_effect(xv), effect, rtol=0, atol=1e-12)
                    breakdown = fitting.groupby(spec.treatment_column)[outcome].agg(["size", "sum"])
                    row = identity | {"policy": policy, "train_n": len(fitting), "full_train_n": len(train),
                        "fit_seconds": elapsed, "model_path": relpath.as_posix(), "training_fingerprint_sha256":
                            fingerprint if fitting is train else _fingerprint(fitting, spec, outcome),
                        "feature_allowlist": list(spec.feature_columns), "train_partition": "train",
                        "evaluation_partition": "validation", "source": source, "seed": seed,
                        "training_arm_counts": {"control": int(breakdown.loc[0, "size"]), "treated": int(breakdown.loc[1, "size"])},
                        "training_arm_events": {"control": int(breakdown.loc[0, "sum"]), "treated": int(breakdown.loc[1, "sum"])},
                        "model_origin": "session_07", "environment": environment, "reload_verified": True,
                        "training_propensity": propensity.value, "propensity_source": propensity.source,
                        "configuration": {"outcome_model": "standardized logistic", "treatment_model": "fixed probability",
                            "folds": 3, "discrete_outcome": True, "discrete_treatment": True,
                            "forest": None if policy == "linear_dml" else {
                                "trees": estimator.trees, "min_leaf": estimator.min_leaf, "honest": True, "jobs": 2,
                                "max_depth": 12, "max_samples": 0.4}},
                        "mean_effect": float(effect.mean()), "effect_min": float(effect.min()),
                        "effect_max": float(effect.max()), "outside_bounds_n": int((np.abs(effect) > 1).sum()),
                        "feature_importance": None if policy == "linear_dml" else dict(zip(
                            spec.feature_columns, map(float, estimator.model_.feature_importances_), strict=True))}
                    advanced[policy] = (estimator, row)
                    comparison_details[policy] = {k: row[k] for k in ("train_n", "fit_seconds", "model_origin")}
                tau = aligned_synthetic_effects(val, spec, truth_path) if truth_path is not None else None
                true_training_ate = None if tau is None else float(aligned_synthetic_effects(train, spec, truth_path).mean())
                structural.append(identity | asdict(pl) | {"train_n": len(train), "fit_seconds": pl_seconds,
                    "target": "constant-propensity structural coefficient", "propensity": propensity.value,
                    "propensity_source": propensity.source, "true_training_mean_cate": true_training_ate})
                scores = {"random": np.zeros(len(val)), "predictive": predicted, **effects}
                measured = {}
                if tau is not None:
                    scores["oracle"] = tau
                    for policy, effect in effects.items():
                        measured[policy] = effect_accuracy(effect, tau)
                        accuracy.append(identity | {"policy": policy, "validation_n": len(val),
                            **comparison_details[policy], **measured[policy]})
                selections, ranking, _, bins = _benchmark(identity, val, spec, outcome, scores, effects, tau, seed)
                ranks.extend(ranking)
                targets.extend(selections)
                calibration.extend(bins)
                for result in ranking:
                    policy = result["policy"]
                    comparisons.append(result | comparison_details.get(policy, {}) | measured.get(policy, {}))
                    if policy in advanced:
                        model, row = advanced[policy]
                        row["validation_metrics"] = result | measured.get(policy, {})
                        manifest.append(row)
                        uncertainty.append(identity | {"policy": policy} | _interval_summary(model, xv, tau, seed))
    confounded, confounded_summary, confounded_diagnostics = confounding_study(
        seed, min(repetitions, 2) if fixture else repetitions, rows=700 if fixture else 2400, estimators=20 if fixture else 55,
    )
    for name, rows in (("dml_estimation_results", structural), ("high_dimensional_confounding", confounded),
                       ("synthetic_effect_accuracy", accuracy), ("uplift_ranking_metrics", ranks),
                       ("targeting_comparison", targets), ("uplift_calibration", calibration),
                       ("causal_forest_uncertainty", [r for r in uncertainty if r["policy"] == "causal_forest"]),
                       ("linear_dml_uncertainty", [r for r in uncertainty if r["policy"] == "linear_dml"]),
                       ("final_causal_ml_comparison", comparisons)):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    write_json(target / "causal_model_manifest.json", {"models": manifest})
    write_json(target / "dml_diagnostics.json", {"benchmarks": diagnostics, "confounding": confounded_diagnostics})
    _plots(target / "causal_ml_comparison.png", accuracy, confounded)
    summary = {"seed": seed, "fixture": fixture, "environment": environment, "models": len(manifest),
        "runtime_seconds": perf_counter() - start, "forest_max_train": forest_max_train,
        "benchmark_evaluation_partitions": ["validation"], "metric_conventions": METADATA,
        "confounding_summary": confounded_summary,
        "synthetic_e_at_20": [r for r in targets if r["dataset"] == "synthetic_E" and r["fraction"] == 0.2]}
    write_json(target / "session_07_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forest-max-train", type=int, default=120_000)
    parser.add_argument("--repetitions", type=int, default=30)
    args = parser.parse_args()
    run(**vars(args))
    print(args.output or args.root / "artifacts" / ("session_07_fixture" if args.fixture else "session_07"))


if __name__ == "__main__":
    main()
