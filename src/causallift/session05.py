"""Benchmark S/T learners and uplift rankings on the fixed development splits."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import joblib
import matplotlib
import numpy as np
import pandas as pd

from causallift.experimentation import analyze_binary
from causallift.meta_learners import MetaLearner
from causallift.prediction import evaluate, fit_model, training_fingerprint
from causallift.session01 import write_json
from causallift.session04 import _inputs
from causallift.targeting import FRACTIONS, aligned_synthetic_effects, rank_scores
from causallift.uplift import RandomizedEvaluation, effect_accuracy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

PAIRS = (("s", "logistic"), ("s", "lightgbm"), ("t", "logistic"), ("t", "lightgbm"))
POLICIES = {"random": "Random", "predictive": "Predictive", "constant_effect": "Constant effect",
            "s_logistic": "S logistic", "s_lightgbm": "S LightGBM",
            "t_logistic": "T logistic", "t_lightgbm": "T LightGBM", "oracle": "Oracle"}
METADATA = {
    "gain": "G(k)=sum(TY/e-(1-T)Y/(1-e)) over top k divided by N",
    "uplift_at_k": "sum of pseudo-outcomes over top k divided by k",
    "auuc": "Trapezoidal integral of G(k) over k/N including origin and endpoint",
    "gain_area_above_random": "Area of G(k)-(k/N)G(N)",
    "qini": "Q(k)=(treated_events-control_events*treated_n/control_n)/N within prefix",
    "qini_coefficient": "Area of Q(k)-(k/N)Q(N); unsupported initial prefixes are omitted",
    "uncertainty": "Pointwise independent-row Wald approximation at fixed scores and propensity. No multiplicity adjustment.",
    "criteo": "Released-benchmark approximation using the validation assignment share. Original propensities are unavailable.",
}


def _predictive_baseline(root, fixture, dataset, spec, train, validation, outcome, seed, previous):
    if fixture:
        model = fit_model("lightgbm", spec, train, validation, outcome, seed=seed, estimators=35)
    else:
        entry = previous[dataset, outcome]
        if (entry["training_fingerprint_sha256"] != training_fingerprint(train, spec, outcome)
                or tuple(entry["feature_allowlist"]) != tuple(spec.feature_columns)):
            raise ValueError("Session 04 baseline provenance does not match Session 05 inputs")
        model = joblib.load(root / "artifacts/session_04" / entry["model_path"])
    return model.predict(validation)


def _fit(target, dataset, spec, train, validation, source, outcome, seed, fixture, environment):
    x_train, x_val = spec.features(train), spec.features(validation)
    t_train, y_train = train[spec.treatment_column], train[outcome]
    fingerprint = hashlib.sha256(bytes.fromhex(training_fingerprint(train, spec, outcome)))
    fingerprint.update(t_train.to_numpy(dtype="int8").tobytes())
    policies, results, manifests = {}, [], []
    for learner, base in PAIRS:
        policy = f"{learner}_{base}"
        start = perf_counter()
        model = MetaLearner(learner, base, seed=seed, estimators=25 if fixture else 100).fit(x_train, t_train, y_train)
        fit_seconds = perf_counter() - start
        potential = model.predict_potential_outcomes(x_val)
        relpath = Path("models") / f"{dataset.lower().replace(' ', '_')}_{outcome}_{policy}.joblib"
        joblib.dump(model, target / relpath, compress=3)
        reloaded = joblib.load(target / relpath).predict_potential_outcomes(x_val)
        np.testing.assert_allclose(reloaded.mu0, potential.mu0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(reloaded.mu1, potential.mu1, rtol=0, atol=1e-14)
        factual = np.where(validation[spec.treatment_column], potential.mu1, potential.mu0)
        detail = {"dataset": dataset, "outcome": outcome, "policy": policy, "learner": learner, "base": base,
                  "fit_seconds": fit_seconds, "validation_n": len(validation),
                  "validation_positives": int(validation[outcome].sum()),
                  "training_control_n": model.arm_stats_["control"]["n"],
                  "training_treated_n": model.arm_stats_["treated"]["n"],
                  "training_control_positives": model.arm_stats_["control"]["positives"],
                  "training_treated_positives": model.arm_stats_["treated"]["positives"],
                  "fallback": json.dumps(model.fallbacks_), "mean_predicted_cate": float(potential.effect.mean()),
                  "factual_log_loss": evaluate(validation[outcome], factual)["log_loss"]}
        policies[policy] = potential.effect
        results.append(detail)
        manifests.append(detail | {"model_path": relpath.as_posix(), "training_fingerprint_sha256": fingerprint.hexdigest(),
                                   "feature_allowlist": list(spec.feature_columns), "configuration": model.model_config,
                                   "train_partition": "train", "evaluation_partition": "validation",
                                   "source": source, "seed": seed, "environment": environment,
                                   "reload_verified": True})
    return policies, results, manifests


def _benchmark(identity, validation, spec, outcome, scores, effects, tau, seed):
    n = len(validation)
    t, y = validation[spec.treatment_column], validation[outcome]
    propensity = spec.assignment_probability
    if propensity is None:
        propensity = float(t.mean())
        source = "released validation share" if spec.source_type == "experiment" else "fixture validation share"
    else:
        source = "known randomized design"
    evaluation = RandomizedEvaluation(t, y, propensity)
    comparison, metrics, coordinates, calibration = [], [], [], []
    for policy, score in scores.items():
        order = rank_scores(score, seed=seed)
        curve = evaluation.curve_from_order(order, truth=tau)
        keys = identity | {"policy": policy}
        measures = keys | curve.metrics() | {"n": n, "positives": int(y.sum()), "treated_positives": int(y[t.eq(1)].sum()),
                                             "control_positives": int(y[t.eq(0)].sum()), "propensity": propensity,
                                             "propensity_source": source}
        capacities = [int(np.floor(fraction * n)) for fraction in FRACTIONS]
        for fraction, k in zip(FRACTIONS, capacities, strict=True):
            selected = order[:k]
            row = evaluation.summarize_selection(selected)
            row["gain"] = float(curve.gain[k])
            row["estimated_incremental_conversions"] = float(curve.gain[k] * n)
            row["mean_predicted_cate"] = float(effects[policy][selected].mean()) if k and policy in effects else None
            selected_tau = tau[selected] if tau is not None and k else None
            comparison.append(keys | {"fraction": fraction,
                                      "propensity_source": source,
                                      "mean_true_cate": None if selected_tau is None else float(selected_tau.mean()),
                                      "expected_incremental_conversions": None if selected_tau is None else float(selected_tau.sum()),
                                      "negative_effect_fraction": None if selected_tau is None else float((selected_tau < 0).mean())}
                              | row)
            measures[f"uplift_at_{round(100 * fraction)}"] = row["uplift"]
        metrics.append(measures)
        valid = np.flatnonzero(np.isfinite(curve.qini[1:]))
        first_qini = int(valid[0] + 1) if valid.size else 0
        for k in np.unique(np.r_[np.linspace(0, n, 101, dtype=int), capacities, first_qini]):
            coordinates.append(keys | {"fraction": float(curve.fraction[k]), "k": int(k),
                                      "gain": float(curve.gain[k]), "random_gain": float(curve.random_gain[k]),
                                      "qini": float(curve.qini[k]) if np.isfinite(curve.qini[k]) else None,
                                      "random_qini": float(curve.random_qini[k]),
                                      "true_gain": None if tau is None else float(curve.true_gain[k])})
        if policy in effects:
            bins = 4 if spec.source_type == "experiment" and outcome == "conversion" else 10
            calibration.extend(keys | {"propensity_source": source} | row
                               for row in evaluation.calibration(effects[policy], bins=bins, truth=tau))
    return comparison, metrics, coordinates, calibration


def _plots(target, coordinates, calibration):
    curves = pd.DataFrame(coordinates)
    bins = pd.DataFrame(calibration)
    pairs = list(dict.fromkeys(zip(curves["dataset"], curves["outcome"], strict=True)))
    interest = [pair for pair in pairs if pair[0] in ("Criteo 1M", "synthetic_E")]
    if not interest:
        interest = pairs
    elif not any(dataset.startswith("Criteo") for dataset, _ in interest):
        interest = [pair for pair in pairs if pair[0] == "Criteo 200k"] + interest
    for filename, measure in (("uplift_curves.png", "gain"), ("qini_curves.png", "qini"),
                              ("uplift_calibration.png", None)):
        fig, axes = plt.subplots(1, len(interest), squeeze=False, figsize=(4.7 * len(interest), 3.7), layout="constrained")
        for ax, (dataset, outcome) in zip(axes[0], interest, strict=True):
            data = bins if measure is None else curves
            data = data.loc[data["dataset"].eq(dataset) & data["outcome"].eq(outcome)]
            ax.set_title(f"{dataset.replace('_', ' ')} · {outcome}")
            if measure is None:
                data = data.dropna(subset=["uplift"])
                for policy, group in data.groupby("policy", sort=False):
                    if policy == "constant_effect":
                        continue
                    ax.errorbar(100 * group["mean_predicted_cate"], 100 * group["uplift"],
                                yerr=196 * group["effect_se"], marker="o", linestyle="none", ms=4,
                                capsize=2, label=POLICIES[policy])
                if not data.empty:
                    limits = [100 * min(data["mean_predicted_cate"].min(), data["uplift"].min()),
                              100 * max(data["mean_predicted_cate"].max(), data["uplift"].max())]
                    ax.plot(limits, limits, "--", color="#777777", label="Ideal")
                ax.set(xlabel="Predicted CATE (pp)", ylabel="Estimated uplift (pp)")
            else:
                for policy, group in data.groupby("policy", sort=False):
                    if policy in ("random", "constant_effect"):
                        continue
                    group = group.dropna(subset=[measure])
                    ax.plot(100 * group["fraction"], 100 * group[measure], label=POLICIES[policy])
                reference = data.loc[data["policy"].eq("random")]
                ax.plot(100 * reference["fraction"], 100 * reference[f"random_{measure}"], "--",
                        color="#777777", label="Random reference")
                if measure == "gain" and data.loc[data["policy"].eq("oracle"), "true_gain"].notna().any():
                    oracle = data.loc[data["policy"].eq("oracle")]
                    ax.plot(100 * oracle["fraction"], 100 * oracle["true_gain"], ":", color="#262626", label="True oracle gain")
                ax.set(xlabel="Eligible population targeted (%)", ylabel=f"Normalized {measure} (×100)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc="best")
        fig.savefig(target / filename, dpi=150)
        plt.close(fig)


def run(root: Path, *, output: Path | None = None, fixture: bool = False,
        development_only: bool = False, seed: int = 42) -> dict:
    start = perf_counter()
    target = output or root / "artifacts" / ("session_05_fixture" if fixture else "session_05")
    (target / "models").mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "dependencies": {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "pyarrow", "matplotlib",
    )}}
    previous = {}
    if not fixture:
        manifest = root / "artifacts/session_04/predictive_model_manifest.json"
        previous = {(r["dataset"], r["outcome"]): r for r in json.loads(manifest.read_text(encoding="utf-8"))["models"]
                    if r["model"] == "lightgbm"}
    results, accuracies, rankings, targeting, bins, coordinates, manifests = ([] for _ in range(7))
    with TemporaryDirectory() as scratch:
        for dataset, spec, train, validation, truth_path, source in _inputs(
            root, Path(scratch), fixture=fixture, development_only=development_only, seed=seed,
        ):
            for outcome in (spec.outcome_column, *spec.secondary_outcomes):
                fitted, details, records = _fit(target, dataset, spec, train, validation, source, outcome,
                                                seed, fixture, environment)
                predictive = _predictive_baseline(root, fixture, dataset, spec, train, validation, outcome, seed, previous)
                n = len(validation)
                ate = analyze_binary(train[spec.treatment_column], train[outcome]).absolute_lift
                effects = {"constant_effect": np.full(n, ate), **fitted}
                scores = {"random": np.zeros(n), "predictive": predictive, **effects}
                identity = {"dataset": dataset, "outcome": outcome}
                tau = None
                if truth_path is not None:
                    tau = aligned_synthetic_effects(validation, spec, truth_path)
                    scores["oracle"] = tau
                    accuracies.extend(identity | {"policy": policy, "n": n, "true_ate": float(tau.mean())}
                                      | effect_accuracy(prediction, tau) for policy, prediction in effects.items())
                policies, metrics, coords, calibration = _benchmark(
                    identity, validation, spec, outcome, scores, effects, tau, seed,
                )
                for record in records:
                    metric = next(row for row in metrics if row["policy"] == record["policy"])
                    accuracy = next((row for row in accuracies
                                     if all(row[key] == record[key] for key in ("dataset", "outcome", "policy"))), None)
                    record["validation_metrics"] = metric | {"synthetic_accuracy": accuracy}
                results.extend(details)
                manifests.extend(records)
                targeting.extend(policies)
                rankings.extend(metrics)
                bins.extend(calibration)
                coordinates.extend(coords)
    for name, rows in (
        ("causal_estimator_results", results), ("synthetic_effect_accuracy", accuracies),
        ("uplift_ranking_metrics", rankings), ("targeting_comparison", targeting),
        ("uplift_calibration", bins), ("uplift_curve_coordinates", coordinates),
    ):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    write_json(target / "causal_model_manifest.json", {"models": manifests})
    _plots(target, coordinates, bins)
    summary = {"seed": seed, "fixture": fixture, "environment": environment, "models": len(manifests),
               "evaluated_partitions": ["validation"], "runtime_seconds": perf_counter() - start,
               "metrics": METADATA, "causal_estimator_results": results, "rankings": rankings,
               "synthetic_accuracy": accuracies,
               "synthetic_e_at_20": [r for r in targeting if r["dataset"] == "synthetic_E" and r["fraction"] == 0.2]}
    write_json(target / "session_05_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--development-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(**vars(args))
    print(args.output or args.root / "artifacts" / ("session_05_fixture" if args.fixture else "session_05"))


if __name__ == "__main__":
    main()
