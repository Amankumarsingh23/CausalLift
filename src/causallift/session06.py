"""Run matched S/T/X/R comparisons and a paired treatment-imbalance study."""

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

from causallift.data.criteo import CRITEO
from causallift.data.splits import split_data
from causallift.experimentation import analyze_binary
from causallift.meta_learners import MetaLearner
from causallift.prediction import training_fingerprint
from causallift.residual_learners import ResidualLearner, training_propensity
from causallift.session01 import write_json
from causallift.session04 import _inputs, _partitions
from causallift.session05 import METADATA, PAIRS, _benchmark, _predictive_baseline
from causallift.simulation import simulate
from causallift.targeting import aligned_synthetic_effects, rank_scores
from causallift.uplift import RandomizedEvaluation, effect_accuracy

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

XR = (("x", "linear"), ("x", "lightgbm"), ("r", "linear"), ("r", "lightgbm"))
STUDY_SEEDS = (101, 202, 303)
STUDY_POLICIES = ("t_lightgbm", "x_lightgbm", "r_lightgbm")


def _cohorts(root, temporary, *, fixture, development_only, seed):
    # Finish 200k and synthetic cohorts before opening the optional larger input.
    yield from _inputs(root, temporary, fixture=fixture, development_only=True, seed=seed)
    relative = "data/processed/benchmark_1m/data/processed/criteo_v2_1.parquet"
    path = root / relative
    if not fixture and not development_only and path.is_file():
        yield "Criteo 1M", CRITEO, *_partitions(path, CRITEO), None, relative


def _manifest(path, field=None, value=None):
    return {
        (entry["dataset"], entry["outcome"], entry.get("policy", entry.get("model"))): entry
        for entry in json.loads(path.read_text(encoding="utf-8"))["models"]
        if field is None or entry.get(field) == value
    }


def _range(effect):
    outside = int((np.abs(effect) > 1).sum())
    return {"raw_effect_min": float(effect.min()), "raw_effect_max": float(effect.max()),
            "outside_bounds_n": outside, "outside_bounds_fraction": outside / len(effect),
            "clipped": False}


def _study() -> list[dict]:
    rows = []
    for seed in STUDY_SEEDS:
        for scenario in ("B", "D"):
            observed, truth, spec = simulate(scenario, seed=seed)
            observed["split"] = split_data(observed, spec, seed)
            train, val = (observed.loc[observed["split"].eq(split)] for split in ("train", "validation"))
            x, xv = spec.features(train), spec.features(val)
            t, y = train[spec.treatment_column], train[spec.outcome_column]
            propensity = training_propensity(spec, train)
            estimates = {}
            for learner, _ in (("t", "lightgbm"), *[pair for pair in XR if pair[1] == "lightgbm"]):
                model = (MetaLearner("t", "lightgbm", seed=seed) if learner == "t"
                         else ResidualLearner(learner, "lightgbm", propensity, seed=seed))
                estimates[f"{learner}_lightgbm"] = model.fit(x, t, y).predict_effect(xv)
            # Align truth only after every learner has produced effects.
            tau = truth.set_index("unit_id").loc[val["unit_id"], "tau_true"].to_numpy()
            evaluation = RandomizedEvaluation(val[spec.treatment_column], val[spec.outcome_column], propensity.value)
            for policy, effect in estimates.items():
                k = len(val) // 5
                selected = rank_scores(effect, seed=seed)[:k]
                rows.append({"study": "repeated", "seed": seed, "scenario": scenario, "policy": policy,
                             "control_train_n": int(t.eq(0).sum()), "treated_train_n": int(t.eq(1).sum()),
                             "propensity": propensity.value, "expected_incremental_conversions_at_20": float(tau[selected].sum()),
                             **effect_accuracy(effect, tau), **evaluation.curve(effect, seed=seed, truth=tau).metrics()})
    return rows


def _plot(target, rows):
    frame = pd.DataFrame(rows)
    fixed, repeated = (frame.loc[frame["study"].eq(mode)] for mode in ("fixed", "repeated"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    policies = [f"{family}_{base}" for family, base in (*PAIRS, *XR)]
    for i, (scenario, color) in enumerate((("B", "#24618e"), ("D", "#a04e2c"))):
        values = fixed.loc[fixed["scenario"].eq(scenario)].set_index("policy")["root_pehe"].reindex(policies)
        axs[0].barh(np.arange(len(policies)) + (i - 0.5) * 0.34, 100 * values, height=0.32,
                    label=f"Scenario {scenario}", color=color)
        for j, policy in enumerate(STUDY_POLICIES):
            points = repeated.loc[repeated["scenario"].eq(scenario) & repeated["policy"].eq(policy), "root_pehe"]
            pos = j + (i - 0.5) * 0.30
            axs[1].scatter(100 * points, np.full(len(points), pos), color=color, alpha=0.65, s=22)
            if len(points):
                axs[1].plot(100 * points.mean(), pos, marker="|", ms=12, color=color)
    axs[0].set(yticks=range(len(policies)), yticklabels=[x.replace("_", " ") for x in policies], title="Fixed validation split")
    axs[1].set(yticks=range(len(STUDY_POLICIES)), yticklabels=[x.replace("_", " ") for x in STUDY_POLICIES],
               title="Repeated seeds · points and means")
    for ax in axs:
        ax.invert_yaxis()
        ax.set_xlabel("Root PEHE (percentage points)")
    axs[0].legend(fontsize=8)
    fig.savefig(target / "imbalance_comparison.png", dpi=150)
    plt.close(fig)


def run(root: Path, *, output: Path | None = None, fixture: bool = False,
        development_only: bool = False, seed: int = 42) -> dict:
    start = perf_counter()
    target = output or root / "artifacts" / ("session_06_fixture" if fixture else "session_06")
    (target / "models").mkdir(parents=True, exist_ok=True)
    environment = {"python": platform.python_version(), "dependencies": {name: version(name) for name in (
        "numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "joblib", "pyarrow", "matplotlib", "threadpoolctl",
    )}}
    previous = {} if fixture else _manifest(root / "artifacts/session_05/causal_model_manifest.json")
    predictive_manifest = {} if fixture else {
        key[:2]: entry for key, entry in _manifest(root / "artifacts/session_04/predictive_model_manifest.json",
                                                   "model", "lightgbm").items()
    }
    comparisons, accuracy, ranking, targeting, calibration, imbalance, models = ([] for _ in range(7))
    crossfit = {}
    with TemporaryDirectory() as temporary:
        for dataset, spec, train, val, truth_path, source in _cohorts(
            root, Path(temporary), fixture=fixture, development_only=development_only, seed=seed,
        ):
            propensity = training_propensity(spec, train)
            xt, xv = spec.features(train), spec.features(val)
            tt, tv = train[spec.treatment_column], val[spec.treatment_column]
            groups = train[spec.unit_id] if spec.unit_id else None
            for outcome in (spec.outcome_column, *spec.secondary_outcomes):
                yt, yv = train[outcome], val[outcome]
                identity = {"dataset": dataset, "outcome": outcome}
                digest = hashlib.sha256(bytes.fromhex(training_fingerprint(train, spec, outcome)))
                digest.update(tt.to_numpy(dtype="int8").tobytes())
                fingerprint = digest.hexdigest()
                baseline = _predictive_baseline(root, fixture, dataset, spec, train, val,
                                                outcome, seed, predictive_manifest)
                ate = analyze_binary(tt, yt).absolute_lift
                effects = {"constant_effect": np.full(len(val), ate)}
                pending = []
                for learner, base in (*PAIRS, *XR):
                    policy = f"{learner}_{base}"
                    key = (dataset, outcome, policy)
                    row = identity | {"policy": policy, "learner": learner, "base": base,
                                      "training_n": len(train), "validation_n": len(val),
                                      "train_treated_n": int(tt.eq(1).sum()), "train_control_n": int(tt.eq(0).sum()),
                                      "train_treated_events": int(yt[tt.eq(1)].sum()),
                                      "train_control_events": int(yt[tt.eq(0)].sum())}
                    if learner in ("s", "t"):
                        if fixture:
                            model = MetaLearner(learner, base, seed=seed, estimators=25).fit(xt, tt, yt)
                        else:
                            prior = previous[key]
                            if (prior["training_fingerprint_sha256"] != fingerprint
                                    or tuple(prior["feature_allowlist"]) != tuple(spec.feature_columns)):
                                raise ValueError(f"Session 05 model provenance differs for {key}")
                            model = joblib.load(root / "artifacts/session_05" / prior["model_path"])
                        effects[policy] = model.predict_effect(xv)
                        row.update(fit_seconds=0.0, model_origin="fixture" if fixture else "session_05")
                    else:
                        then = perf_counter()
                        model = ResidualLearner(learner, base, propensity, seed=seed, estimators=20 if fixture else 65)
                        model.fit(xt, tt, yt, groups=groups)
                        row.update(fit_seconds=perf_counter() - then, model_origin="session_06",
                                   fit_propensity=propensity.value, fit_propensity_source=propensity.source)
                        effect = model.predict_effect(xv)
                        effects[policy] = effect
                        candidate = Path("models") / f"{dataset.lower().replace(' ', '_')}_{outcome}_{policy}.joblib"
                        joblib.dump(model, target / candidate, compress=3)
                        np.testing.assert_allclose(joblib.load(target / candidate).predict_effect(xv), effect, rtol=0, atol=1e-14)
                        row.update(_range(effect))
                        if learner == "r":
                            loss = model.validation_r_loss(xv, tv, yv, effect)
                            constant_loss = model.validation_r_loss(xv, tv, yv, effects["constant_effect"])
                            row["validation_r_loss"] = loss
                            crossfit["/".join(key)] = identity | {"policy": policy, "training_n": len(train), "seed": seed,
                                "propensity_source": propensity.source, "fit_seconds": row["fit_seconds"],
                                "validation_r_loss": loss, "constant_effect_r_loss": constant_loss, **model.diagnostics_}
                        pending.append(row | {"model_path": candidate.as_posix(), "source": source,
                                              "train_partition": "train", "evaluation_partition": "validation",
                                              "feature_allowlist": list(spec.feature_columns), "seed": seed,
                                              "training_fingerprint_sha256": fingerprint, "reload_verified": True,
                                              "configuration": model.model_config, "environment": environment,
                                              "outcome_fallbacks": model.diagnostics_.get("outcome_fallbacks",
                                                  {str(fold["fold"]): fold["constant_fallback"]
                                                   for fold in model.diagnostics_.get("folds", [])
                                                   if fold["constant_fallback"] is not None})})
                    comparisons.append(row)
                tau = aligned_synthetic_effects(val, spec, truth_path) if truth_path is not None else None
                scores = {"random": np.zeros(len(val)), "predictive": baseline, **effects}
                if tau is not None:
                    scores["oracle"] = tau
                targets, ranks, _, cal = _benchmark(identity, val, spec, outcome, scores, effects, tau, seed)
                targeting.extend(targets)
                ranking.extend(ranks)
                calibration.extend(cal)
                for record in pending:
                    record["validation_metrics"] = next(r for r in ranks if r["policy"] == record["policy"])
                    if tau is not None:
                        record["synthetic_accuracy"] = effect_accuracy(effects[record["policy"]], tau)
                    models.append(record)
                if tau is not None:
                    for policy, effect in effects.items():
                        measured = identity | {"policy": policy, "n": len(val), "true_ate": float(tau.mean()),
                                               **effect_accuracy(effect, tau)}
                        accuracy.append(measured)
                        if dataset in ("synthetic_B", "synthetic_D"):
                            rank = next(r for r in ranks if r["policy"] == policy)
                            top = next(r for r in targets if r["policy"] == policy and r["fraction"] == 0.2)
                            imbalance.append(measured | {"study": "fixed", "scenario": dataset[-1], "seed": seed,
                                "control_train_n": int(tt.eq(0).sum()), "treated_train_n": int(tt.eq(1).sum()),
                                "propensity": propensity.value, "qini_coefficient": rank["qini_coefficient"],
                                "auuc": rank["auuc"], "true_auuc": rank["true_auuc"],
                                "expected_incremental_conversions_at_20": top["expected_incremental_conversions"]})
    if not fixture:
        imbalance.extend(_study())
        _plot(target, imbalance)
    for name, rows in (
        ("meta_learner_comparison", comparisons), ("synthetic_effect_accuracy", accuracy), ("uplift_ranking_metrics", ranking),
        ("targeting_comparison", targeting), ("uplift_calibration", calibration), ("imbalance_study", imbalance),
    ):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    write_json(target / "crossfit_diagnostics.json", {"runs": crossfit})
    write_json(target / "causal_model_manifest.json", {"models": models})
    summary = {"seed": seed, "fixture": fixture, "runtime_seconds": perf_counter() - start,
               "environment": environment, "evaluated_partitions": ["validation"], "models": len(models),
               "study_seeds": [] if fixture else list(STUDY_SEEDS), "metric_conventions": METADATA,
               "fit_propensity_note": "Known synthetic design; released training share for Criteo.",
               "effect_prediction_policy": "Unclipped regression effects.",
               "synthetic_e_at_20": [r for r in targeting if r["dataset"] == "synthetic_E" and r["fraction"] == 0.2]}
    write_json(target / "session_06_summary.json", summary)
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
    print(args.output or args.root / "artifacts" / ("session_06_fixture" if args.fixture else "session_06"))


if __name__ == "__main__":
    main()
