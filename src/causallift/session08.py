"""Generate classical identification and train-only propensity diagnostics."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

from causallift.causal_dags import CausalDAG
from causallift.confounding_demo import simulate_confounding
from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import CRITEO
from causallift.data.splits import split_data
from causallift.double_ml import partially_linear_dml
from causallift.experiment_health import covariate_balance
from causallift.prediction import calibration_bins, evaluate, training_fingerprint
from causallift.propensity import fit_assignments, overlap, propensity_prediction
from causallift.session01 import write_json
from causallift.simulation import simulate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def graph_examples():
    return {
        "confounder": CausalDAG(("X", "T", "Y"), (("X", "T"), ("X", "Y"), ("T", "Y")),
                                "T", "Y", ("X",), adjustment=("X",)),
        "mediator": CausalDAG(("T", "M", "Y"), (("T", "M"), ("M", "Y"), ("T", "Y")), "T", "Y", ("M",)),
        "collider": CausalDAG(("T", "C", "U", "Y"), (("T", "C"), ("U", "C"), ("U", "Y"), ("T", "Y")),
                              "T", "Y", ("C",), ("U",)),
        "hidden": CausalDAG(("U", "T", "Y"), (("U", "T"), ("U", "Y"), ("T", "Y")), "T", "Y", latent=("U",)),
    }


def bias_demonstrations(seed, n):
    rng = np.random.default_rng(seed)
    x, noise, mechanism = rng.normal(size=(3, n))
    randomized = rng.binomial(1, .5, n)
    confounded = rng.binomial(1, .2 + .6 * (x > 0))
    mediator = 2 * randomized + mechanism
    collider = randomized + x + mechanism
    rows = []
    for case, t, y, adj, true in (
        ("confounder", confounded, confounded + 2 * x + noise, x, 1.),
        ("mediator", randomized, randomized + 1.5 * mediator + noise, mediator, 4.),
        ("collider", randomized, randomized + 2 * x + noise, collider, 1.),
    ):
        for name, columns in (("unadjusted", [np.ones(n), t]), ("adjusted", [np.ones(n), t, adj])):
            coef = float(np.linalg.lstsq(np.column_stack(columns), y, rcond=None)[0][1])
            rows.append({"case": case, "analysis": name, "coefficient": coef, "true_total_effect": true,
                         "error_relative_to_total": coef - true, "n": n})
    return rows


def _inputs(root, fixture, seed):
    for scenario in "ACF":
        source = f"data/synthetic/synthetic_{scenario}.parquet"
        path = root / source
        if fixture:
            observed, truth, spec = simulate(scenario, 800, seed)
            observed["split"] = split_data(observed, spec, seed)
            source = "generated offline fixture"
        else:
            spec = DatasetSpec(**json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8")))
            observed = None
            truth = path.with_stem(path.stem + "_truth")
        train, val = (load_dataset(path, spec, split) if observed is None else observed.loc[observed.split.eq(split)]
                      for split in ("train", "validation"))
        yield scenario, source, spec, train, val, truth
    if not fixture:
        source = "data/processed/criteo_v2_1.parquet"
        train, val = (load_dataset(root / source, CRITEO, split) for split in ("train", "validation"))
        yield "Criteo 200k", source, CRITEO, train, val, None


def _nuisance(root, fixture, seed):
    path = root / "artifacts/session_07/dml_diagnostics.json"
    original = [] if fixture else json.loads(path.read_text(encoding="utf-8"))["confounding"][:3]
    results = []
    seeds = [seed] if fixture else [item["seed"] for item in original]
    for i, draw in enumerate(seeds):
        X, t, y, truth = simulate_confounding(draw, rows=800 if fixture else 2400)
        estimate, detail = partially_linear_dml(X, t, y, binary_outcome=False, seed=draw, estimators=55)
        comparison = {}
        for model in ("logistic", "lightgbm"):
            _, cross, _ = fit_assignments(X, t, model, seed=draw)
            p = cross.predictions
            comparison[model] = evaluate(t, p) | {"rmse_against_design": float(np.sqrt(np.mean((p-truth)**2))),
                "overlap": overlap(t, p), "calibration": calibration_bins(t, p),
                "treatment_residual_sd": float((t-p).std())}
        results.append({"seed": draw, "rows": len(t), "original_estimate": None if fixture else original[i]["estimate"],
                        "recomputed_estimate": estimate.estimate, "outcome_oof_rmse": detail["outcome_metrics"]["rmse"],
                        "known_outcome_noise_sd": 1.0, "propensity_models": comparison})
    return {"draws": results, "session07_coverage": None if fixture else "0/30 in the original simulation",
            "interpretation": "Finite-sample nuisance error and nonlinear misspecification are plausible contributors; the comparison cannot isolate a sole cause."}


def _plots(target, graphs, balance, curves):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3), layout="constrained")
    positions = [
        {"X": (0,1), "T": (-1,0), "Y": (1,0)}, {"M": (0,1), "T": (-1,0), "Y": (1,0)},
        {"C": (0,1), "U": (1,1), "T": (-1,0), "Y": (1,0)}, {"U": (0,1), "T": (-1,0), "Y": (1,0)},
    ]
    for ax, (name, spec), pos in zip(axes, graphs.items(), positions, strict=True):
        nx.draw_networkx(spec.graph(), pos=pos, ax=ax, node_color=["#e6a683" if n in spec.latent else "#a7cde6"
                             for n in spec.nodes], node_size=650, arrowsize=19)
        ax.set_title(name.capitalize())
        ax.set_axis_off()
    fig.savefig(target / "causal_graphs.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, layout="constrained")
    for ax, (name, rows) in zip(axes, pd.DataFrame(balance).groupby("dataset", sort=False), strict=True):
        ax.scatter(rows.smd, rows.feature, color="#226992")
        for x in (-.1, 0, .1):
            ax.axvline(x, color="gray", linestyle="--", linewidth=.8)
        ax.set(title=name, xlabel="Unadjusted SMD")
    fig.savefig(target / "covariate_balance.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(curves), figsize=(3.1 * len(curves), 3.3), squeeze=False, layout="constrained")
    for ax, (name, (t, p)) in zip(axes[0], curves.items(), strict=True):
        for arm, label in [(0, "Control"), (1, "Treated")]:
            ax.hist(p[t == arm], bins=np.linspace(0,1,21), weights=np.full((t == arm).sum(), 1/(t == arm).sum()),
                    histtype="step", label=label)
        ax.set(title=name, xlabel="Assignment probability", xlim=(0,1))
    axes[0,0].set_ylabel("Fraction within arm per bin")
    axes[0,0].legend(fontsize=8)
    fig.savefig(target / "propensity_overlap.png", dpi=140)
    plt.close(fig)


def run(root, *, fixture=False, output=None, seed=42):
    start = perf_counter()
    target = output or root / "artifacts" / ("session_08_fixture" if fixture else "session_08")
    models = target / "models"
    models.mkdir(parents=True, exist_ok=True)
    graphs = graph_examples()
    identification = {name: {"empty": dag.backdoor(()), "proposed": dag.backdoor()}
                      for name, dag in graphs.items()}
    for case, variable in (("mediator","M"), ("collider","C"), ("hidden","U")):
        identification[case]["conditioned"] = graphs[case].backdoor((variable,))
    collider_path = graphs["collider"].graph()
    collider_path.remove_edge("T", "Y")
    identification["collider"]["path_demo_without_direct_effect_edge"] = {
        "unconditioned_d_separated": nx.is_d_separator(collider_path, {"T"}, {"Y"}, set()),
        "conditioned_d_separated": nx.is_d_separator(collider_path, {"T"}, {"Y"}, {"C"}),
    }
    metrics, bins, support, balance, manifest = ([] for _ in range(5))
    curves = {}
    for scenario, source, spec, train, val, truth in _inputs(root, fixture, seed):
        column = spec.treatment_column
        assignment = train[column]
        tv = val[column].to_numpy()
        xv = spec.features(val)
        methods = {}
        aligned = None
        if scenario != "Criteo 200k":
            balance.extend(covariate_balance(val, spec, spec.feature_columns))
        for method in ("constant", "logistic", "lightgbm"):
            model, cross, config = fit_assignments(spec.features(train), assignment, method, seed=seed,
                    groups=train[spec.unit_id] if spec.unit_id is not None else None)
            p = propensity_prediction(model, xv)
            methods[method] = p
            stem = scenario.replace(" ", "_") + "_" + method
            model_path, row_path = models / (stem + ".joblib"), models / (stem + ".parquet")
            joblib.dump(model, model_path, compress=3)
            np.testing.assert_allclose(propensity_prediction(joblib.load(model_path), xv), p)
            pd.concat([pd.DataFrame({"partition": part, "position": np.arange(len(frame)),
                           "unit_id": frame[spec.unit_id].to_numpy() if spec.unit_id else None,
                           "fold": fold, "propensity": values})
                       for part, frame, fold, values in (
                         ("train", train, cross.membership, cross.predictions), ("validation", val, -1, p),
                       )], ignore_index=True).to_parquet(row_path, index=False)
            manifest.append({"scenario": scenario, "model": method, "source": source,
                             "model_path": model_path.relative_to(target).as_posix(),
                             "prediction_path": row_path.relative_to(target).as_posix(),
                             "row_identity": spec.unit_id or "position within unchanged ordered partition",
                             "train_fingerprint": training_fingerprint(train, spec, column),
                             "validation_fingerprint": training_fingerprint(val, spec, column),
                             "feature_allowlist": list(spec.feature_columns), "target": column,
                             "configuration": config, "seed": seed, "fold_strategy": cross.strategy,
                             "folds": list(cross.folds), "warnings": list(cross.warnings),
                             "oof_metrics": evaluate(assignment, cross.predictions)})
        if truth is not None:
            if isinstance(truth, Path):
                truth = pd.read_parquet(truth, columns=["unit_id", "propensity_true"])
            aligned = truth.set_index("unit_id").loc[val[spec.unit_id], "propensity_true"].to_numpy()
        for method, p in methods.items():
            row = {"scenario": scenario, "model": method, **evaluate(tv, p),
                   "true_propensity_rmse": None, "true_propensity_mae": None}
            if scenario == "C":
                row.update(true_propensity_rmse=float(np.sqrt(np.mean((p-aligned)**2))),
                           true_propensity_mae=float(np.abs(p-aligned).mean()))
            metrics.append(row)
            bins.extend({"scenario": scenario, "model": method, **r} for r in calibration_bins(tv, p))
            support.append({"scenario": scenario, "model": method, **overlap(tv, p)})
        if aligned is not None:
            label = "true_full_information" if scenario == "F" else "known_design" if scenario == "A" else "true_observed_covariates"
            curves[f"{scenario} · {label.replace('_', ' ')}"] = (tv, aligned)
            support.append({"scenario": scenario, "model": label, **overlap(tv, aligned)})
        if scenario in ("C", "F"):
            curves[f"{scenario} · Logistic"] = (tv, methods["logistic"])

    for filename, rows in (("causal_bias_demonstrations", bias_demonstrations(seed, 4000 if fixture else 50000)),
                           ("propensity_model_results", metrics), ("propensity_calibration", bins),
                           ("propensity_overlap_summary", support), ("covariate_balance", balance)):
        pd.DataFrame(rows).to_csv(target / f"{filename}.csv", index=False)
    write_json(target / "causal_graphs.json", {name: asdict(dag) for name, dag in graphs.items()})
    write_json(target / "identification_results.json", identification)
    write_json(target / "propensity_predictions_manifest.json", {
        "models": manifest, "dependencies": {name: version(name) for name in
            ("numpy", "pandas", "scikit-learn", "lightgbm", "networkx", "joblib", "pyarrow")},
    })
    write_json(target / "nuisance_quality_diagnostics.json", _nuisance(root, fixture, seed))
    _plots(target, graphs, balance, curves)
    summary = {"seed": seed, "fixture": fixture, "runtime_seconds": perf_counter()-start, "models": len(manifest),
               "networkx": version("networkx"), "evaluation_partition": "validation",
               "hidden_propensity_note": "F truth conditions on observed features and hidden U; model scores condition on observable features only.",
               "criteo_note": "Released assignment prediction is diagnostic; original trial assignment probabilities are unavailable."}
    write_json(target / "session_08_summary.json", summary)
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
