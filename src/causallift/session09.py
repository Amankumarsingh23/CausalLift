"""Benchmark observational adjustment with held-out and cross-fitted nuisances."""

import argparse
import json
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causallift.crossfit import crossfit_outcome
from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import checksum
from causallift.data.splits import split_data
from causallift.experimentation import analyze_continuous
from causallift.meta_learners import (
    MetaLearner,
    fit_binary_outcome,
    outcome_configuration,
)
from causallift.observational import (
    aipw,
    effective_n,
    ht,
    match,
    overlap_profiles,
    standardize,
    weighted_contrast,
    weighted_smd,
    weights,
)
from causallift.prediction import training_fingerprint
from causallift.propensity import fit_assignments, propensity_prediction
from causallift.session01 import write_json
from causallift.simulation import simulate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

NSW_URL = "https://users.nber.org/~rdehejia/data/"
NSW_FILES = {
    "nswre74_treated": (
        185,
        1,
        "e7b742fe0ff07a0f45e129b4ff108bb9611cd83d53604732c48a8a0a3e20eda3",
    ),
    "nswre74_control": (
        260,
        0,
        "a1364cea459d953dc691a667d99194b4ad335d6d550354fe23a5d2dc58d729b5",
    ),
    "cps_controls": (
        15992,
        0,
        "dce640066f8bcdf447864b158b2cb0390bd3f8d6cbe3a27576a06926477d51f8",
    ),
}
NSW_FEATURES = (
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
)


def fetch_nsw(root):
    folder = root / "data/raw/lalonde"
    folder.mkdir(parents=True, exist_ok=True)
    for name, (_, _, digest) in NSW_FILES.items():
        path = folder / f"{name}.txt"
        if path.exists():
            continue
        with urlopen(NSW_URL + path.name, timeout=30) as response:
            content = response.read()
        import hashlib

        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"Unexpected content from publisher for {path.name}")
        with path.open("xb") as output:
            output.write(content)


def load_saved_propensity(folder, scenario, spec, train, val, seed, model="logistic"):
    models = json.loads(
        (folder / "propensity_predictions_manifest.json").read_text(encoding="utf-8")
    )["models"]
    (row,) = [
        r for r in models if r["scenario"] == scenario and r["model"] == model
    ]
    col = spec.treatment_column
    if (
        row["target"] != col
        or tuple(row["feature_allowlist"]) != tuple(spec.feature_columns)
        or row["configuration"] != outcome_configuration(model, seed, 55)
        or row["seed"] != seed
        or row["row_identity"] != spec.unit_id
        or row["source"] != f"data/synthetic/synthetic_{scenario}.parquet"
    ):
        raise ValueError("Saved propensity configuration or dataset identity differs")
    saved = pd.read_parquet(folder / row["prediction_path"])
    if len(saved) != len(train) + len(val) or set(saved.partition) != {
        "train",
        "validation",
    }:
        raise ValueError("Saved propensity partitions differ")
    aligned = []
    for name, partition in (("train", train), ("validation", val)):
        table = saved.loc[saved.partition.eq(name)]
        if (
            not partition["split"].eq(name).all()
            or row[f"{name}_fingerprint"] != training_fingerprint(partition, spec, col)
            or not np.array_equal(table.position, np.arange(len(partition)))
            or not np.array_equal(table.unit_id, partition[spec.unit_id])
        ):
            raise ValueError("Saved score alignment or fingerprint differs")
        aligned.append(table)
    if (
        set(aligned[0].fold) != {0, 1, 2}
        or not aligned[1].fold.eq(-1).all()
        or not saved.propensity.between(0, 1).all()
    ):
        raise ValueError("Invalid saved folds or scores")
    return (
        aligned[0].propensity.to_numpy(),
        aligned[1].propensity.to_numpy(),
        aligned[0].fold.to_numpy(),
    )


def _outcome_oof(train, spec, seed):
    x = spec.features(train).reset_index(drop=True)
    t, y = (
        train[spec.treatment_column].to_numpy(),
        train[spec.outcome_column].to_numpy(),
    )
    groups = train[spec.unit_id] if spec.unit_id else None
    result = []
    for arm in (0, 1):

        def fit(fold_features, labels):
            positions = fold_features.index.to_numpy()
            keep = labels == arm
            return fit_binary_outcome(
                fold_features.iloc[keep],
                y[positions[keep]],
                outcome_configuration("logistic", seed, 55),
                tuple(x.columns),
            )

        result.append(crossfit_outcome(x, t, t, fit, groups=groups, seed=seed))
    if not np.array_equal(result[0].membership, result[1].membership):
        raise ValueError("Arm-specific nuisance folds differ")
    return result[0].predictions, result[1].predictions, result[0].membership


def _result(dataset, method, estimand, population, n, estimate, truth, **extra):
    error = None if truth is None or estimate is None else estimate - truth
    return {
        "dataset": dataset,
        "method": method,
        "estimand": estimand,
        "population": population,
        "n": n,
        "estimate": estimate,
        "truth": truth,
        "signed_error": error,
        "absolute_error": None if error is None else abs(error),
        **extra,
    }


def _diagnostic(dataset, profile, kind, t, w):
    rows = []
    for arm, mask in (
        ("all", np.ones(len(t), dtype=bool)),
        ("control", ~t),
        ("treated", t),
    ):
        a = w[mask]
        q = np.quantile(a, [0, 0.5, 0.9, 0.95, 0.99, 1])
        ess = effective_n(a)
        rows.append(
            {
                "dataset": dataset,
                "profile": profile,
                "weight": kind,
                "arm": arm,
                "n": len(a),
                "ess": ess,
                "ess_fraction": ess / len(a),
                "total": float(a.sum()),
                "mean": float(a.mean()),
                "warning": "Max weight >20 or ESS below half"
                if q[-1] > 20 or ess < len(a) / 2
                else "",
                **{
                    f"q{pct}": float(v)
                    for pct, v in zip((0, 50, 90, 95, 99, 100), q, strict=True)
                },
            }
        )
    return rows


def _estimates(dataset, X, t, y, e, mu0, mu1, tau, *, identification, legacy_ties=False):
    t = np.asarray(t, dtype=bool)

    def truth(mask):
        return None if tau is None or not mask.any() else float(tau[mask].mean())

    all_rows = np.ones(len(t), dtype=bool)
    label = "validation"
    w = weights(t, e)
    aw = weights(t, e, "att")
    naive = weighted_contrast(t, y, np.ones(len(t)))
    ipw = weighted_contrast(t, y, w)
    dr = aipw(t, y, e, mu0, mu1)
    warning = f"{identification} Approximate independent-row intervals require adequate nuisance quality and overlap."
    rows = [
        _result(
            dataset, name, estimand, label, n, v, truth(mask), warning=warning, **extras
        )
        for name, estimand, n, v, mask, extras in (
            ("naive_contrast", "ATE comparison", len(t), naive, all_rows, {}),
            ("regression", "ATE", len(t), standardize(mu0, mu1), all_rows, {}),
            ("horvitz_thompson", "ATE", len(t), ht(t, y, e), all_rows, {}),
            ("hajek", "ATE", len(t), ipw, all_rows, {}),
            (
                "aipw",
                "ATE",
                len(t),
                dr["estimate"],
                all_rows,
                {k: v for k, v in dr.items() if k != "estimate"},
            ),
            ("regression", "ATT", int(t.sum()), standardize(mu0, mu1, t), t, {}),
            ("att_weighted", "ATT", int(t.sum()), weighted_contrast(t, y, aw), t, {}),
        )
    ]
    ti, ci, dist, radius = match(t, e, legacy_ties=legacy_ties)
    point = float((y[ti] - y[ci]).mean()) if len(ti) else None
    rows.append(
        _result(
            dataset,
            "matching",
            "matched ATT",
            "caliper-matched treated validation rows",
            len(ti),
            point,
            None if tau is None or not len(ti) else float(tau[ti].mean()),
            warning=identification + " Formal matching inference unsupported.",
        )
    )
    mw = np.bincount(np.r_[ti, ci], minlength=len(t))
    match_row = {
        "dataset": dataset,
        "treated_n": int(t.sum()),
        "control_n": int((~t).sum()),
        "matched_treated": len(ti),
        "unique_controls": len(np.unique(ci)),
        "max_control_reuse": int(np.bincount(ci).max()) if len(ci) else 0,
        "control_ess": effective_n(np.bincount(ci)) if len(ci) else None,
        "reused_control_matches": len(ci) - len(np.unique(ci)),
        "excluded_treated": int(t.sum()) - len(ti),
        "match_rate": len(ti) / t.sum(),
        "logit_caliper": radius,
        "mean_distance": float(dist.mean()) if len(ti) else None,
        "max_distance": float(dist.max()) if len(ti) else None,
        "estimate": point,
        "formal_inference": "unsupported",
    }
    balance, sensitivity, diagnostics = [], [], []
    for name in X.columns:
        x = X[name].to_numpy()
        sd = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
        for method, weight in (
            ("unadjusted", np.ones(len(t))),
            ("matched", mw),
            ("ate_weighted", w),
            ("att_weighted", aw),
        ):
            smd = weighted_smd(t, x, weight, denominator=sd)
            balance.append(
                {
                    "dataset": dataset,
                    "feature": name,
                    "method": method,
                    "smd": smd,
                    "absolute_smd": None if smd is None else abs(smd),
                }
            )
    for profile, (mask, prob) in overlap_profiles(e).items():
        ts, ys, es = t[mask], y[mask], prob[mask]
        if not ts.any() or ts.all():
            sensitivity.append(
                {
                    "dataset": dataset,
                    "profile": profile,
                    "n": int(mask.sum()),
                    "excluded": int((~mask).sum()),
                    "truth": truth(mask),
                    "warning": "Both arms are required",
                }
            )
            continue
        for kind in ("ate", "att", "stabilized"):
            diagnostics.extend(
                _diagnostic(dataset, profile, kind, ts, weights(ts, es, kind))
            )
        estimate = aipw(ts, ys, es, mu0[mask], mu1[mask])["estimate"]
        sensitivity.append(
            {
                "dataset": dataset,
                "profile": profile,
                "n": int(mask.sum()),
                "excluded": int((~mask).sum()),
                "clipped_scores": int(np.sum(prob != e)),
                "target": "restricted ATE" if not mask.all() else "validation ATE",
                "truth": truth(mask),
                "hajek": weighted_contrast(ts, ys, weights(ts, es)),
                "aipw": estimate,
            }
        )
    return rows, match_row, balance, sensitivity, diagnostics


def robustness(seed, reps=80, n=1800):
    rng = np.random.default_rng(seed)
    out = []
    for rep in range(reps):
        x = rng.choice([-1, 1], n)
        e = 0.5 + 0.25 * x
        m0 = 0.35 + 0.15 * x
        m1 = m0 + 0.10
        t = rng.binomial(1, e)
        y = rng.binomial(1, np.where(t, m1, m0))
        for label, p, a, b in (
            ("both_correct", e, m0, m1),
            ("propensity_only", e, np.full(n, 0.35), np.full(n, 0.45)),
            ("outcome_only", 0.5, m0, m1),
            ("neither", 0.5, np.full(n, 0.35), np.full(n, 0.45)),
        ):
            r = aipw(t, y, p, a, b)
            out.append(
                {
                    "rep": rep,
                    "specification": label,
                    "n": n,
                    "true_ate": 0.10,
                    **r,
                    "error": r["estimate"] - 0.10,
                    "covers": r["ci_low"] <= 0.10 <= r["ci_high"],
                }
            )
    return out


def _nsw(root, seed, *, legacy_ties=False):
    folder = root / "data/raw/lalonde"
    if not all((folder / f"{name}.txt").is_file() for name in NSW_FILES):
        return (
            [],
            [],
            [],
            [],
            {"status": "files unavailable; run --download", "source": NSW_URL},
        )
    frames = {}
    for name, (count, arm, digest) in NSW_FILES.items():
        path = folder / f"{name}.txt"
        if checksum(path) != digest:
            raise ValueError(f"Unexpected publisher checksum: {name}")
        data = pd.read_csv(path, sep=r"\s+", header=None)
        if (
            data.shape != (count, 10)
            or not data[0].eq(arm).all()
            or not np.isfinite(data).all().all()
        ):
            raise ValueError(f"Unexpected cohort contents: {name}")
        data.columns = ("treatment", *NSW_FEATURES, "re78")
        frames[name] = data
    pool = pd.concat(
        [frames["nswre74_treated"], frames["cps_controls"]], ignore_index=True
    )
    train, val = train_test_split(
        pool, test_size=0.4, stratify=pool.treatment, random_state=seed
    )
    x, xv = train[list(NSW_FEATURES)], val[list(NSW_FEATURES)]
    t, tv = (
        train.treatment.to_numpy().astype(bool),
        val.treatment.to_numpy().astype(bool),
    )
    y = val.re78.to_numpy()
    nuisance = fit_binary_outcome(
        x, t, outcome_configuration("logistic", seed, 55), NSW_FEATURES
    )
    e = propensity_prediction(nuisance, xv)
    a, b = [
        make_pipeline(StandardScaler(), Ridge(alpha=1))
        .fit(x.loc[mask], train.re78.loc[mask])
        .predict(xv)
        for mask in (~t, t)
    ]
    label = "NSW–CPS"
    rows, matching, balance, sensitive, diagnostic = _estimates(
        label,
        xv,
        tv,
        y,
        e,
        a,
        b,
        None,
        identification="Observational comparability and overlap are unverified.",
        legacy_ties=legacy_ties,
    )
    dr = aipw(tv, y, e, a, b, att=True)
    result = [
        _result(
            label,
            "naive_contrast",
            "ATT comparison",
            "held-out observational",
            len(y),
            weighted_contrast(tv, y, np.ones(len(y))),
            None,
        )
    ]
    result.extend(row for row in rows if row["estimand"] in ("ATT", "matched ATT"))
    result.append(
        _result(
            label,
            "aipw_att",
            "ATT",
            "held-out observational",
            int(tv.sum()),
            dr.pop("estimate"),
            None,
            **dr,
            warning="Approximate independent-row interval; exchangeability and overlap unverified.",
        )
    )
    controls = frames["nswre74_control"].re78.to_numpy()
    ref = analyze_continuous(
        np.r_[np.ones(tv.sum()), np.zeros(len(controls))], np.r_[y[tv], controls]
    )
    result.append(
        _result(
            "NSW experiment",
            "randomized_reference",
            "randomized reference contrast",
            "same held-out treated and 260 randomized controls",
            int(tv.sum() + len(controls)),
            ref.absolute_lift,
            None,
            ci_low=ref.confidence_interval[0],
            ci_high=ref.confidence_interval[1],
        )
    )
    return (
        result,
        [matching],
        balance,
        diagnostic,
        {
            "status": "completed",
            "source": NSW_URL,
            "training_n": len(train),
            "validation_n": len(val),
            "validation_treated": int(tv.sum()),
            "files": {k: {"rows": v[0], "sha256": v[2]} for k, v in NSW_FILES.items()},
            "sensitivity": sensitive,
        },
    )


def _plots(target, balance, ate):
    data = pd.DataFrame(balance)
    c = data.loc[data.dataset.eq("C")]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for method, group in c.groupby("method", sort=False):
        ax.scatter(group.absolute_smd, group.feature, label=method.replace("_", " "))
    ax.axvline(0.1, linestyle="--", color="gray")
    ax.set(xlabel="Absolute standardized mean difference", title="Scenario C · balance")
    ax.legend(fontsize=8)
    fig.savefig(target / "covariate_balance_love_plot.png", dpi=140)
    plt.close(fig)
    data = pd.DataFrame(ate)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharex=True, layout="constrained")
    for ax, (name, group) in zip(
        axes, data.loc[data.population.eq("validation")].groupby("dataset"), strict=True
    ):
        ax.scatter(
            100 * group.estimate, group.method.str.replace("_", " "), color="#245f8a"
        )
        ax.axvline(100 * group.truth.iloc[0], linestyle="--", color="#ab623c")
        ax.set(title=f"Scenario {name}", xlabel="Estimate (pp)")
    fig.savefig(target / "observational_estimator_comparison.png", dpi=140)
    plt.close(fig)


def run(root, *, fixture=False, output=None, seed=42):
    start = perf_counter()
    target = output or root / "artifacts" / (
        "session_09_fixture" if fixture else "session_09"
    )
    target.mkdir(parents=True, exist_ok=True)
    ate, att, matching, balance, sensitivity, diagnostics = [], [], [], [], [], []
    provenance = {}
    for scenario in "ACF":
        if fixture:
            frame, truth, spec = simulate(scenario, 800, seed)
            frame["split"] = split_data(frame, spec, seed)
            train, val = [
                frame.loc[frame.split.eq(part)] for part in ("train", "validation")
            ]
        else:
            path = root / f"data/synthetic/synthetic_{scenario}.parquet"
            spec = DatasetSpec(
                **json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8"))
            )
            train, val = [
                load_dataset(path, spec, part) for part in ("train", "validation")
            ]
        tx = spec.features(train)
        x = spec.features(val)
        tt, ty = train.treatment.to_numpy(), train.outcome.to_numpy()
        t, y = val.treatment.to_numpy(), val.outcome.to_numpy()
        if scenario == "A":
            pe, pv = (
                np.full(len(train), spec.assignment_probability),
                np.full(len(val), spec.assignment_probability),
            )
            fold = None
            source = "known randomized design"
        elif fixture:
            propensity, cross, _ = fit_assignments(
                tx, tt, "logistic", seed=seed, groups=train.unit_id
            )
            pe, pv, fold = (
                cross.predictions,
                propensity_prediction(propensity, x),
                cross.membership,
            )
            source = "fixture logistic"
        else:
            pe, pv, fold = load_saved_propensity(
                root / "artifacts/session_08", scenario, spec, train, val, seed
            )
            source = "verified Session 08 logistic"
        preds = {
            base: MetaLearner("t", base, seed=seed, estimators=55)
            .fit(tx, tt, ty)
            .predict_potential_outcomes(x)
            for base in ("logistic", "lightgbm")
        }
        a, b, outcome_folds = _outcome_oof(train, spec, seed)
        if fold is not None and not np.array_equal(fold, outcome_folds):
            raise ValueError("Assignment and arm-outcome OOF folds differ")
        if not fixture:
            truth = pd.read_parquet(
                path.with_stem(path.stem + "_truth"),
                columns=["unit_id", "tau_true", "propensity_true"],
            )
        indexed = truth.set_index("unit_id")
        tau = indexed.loc[val.unit_id, "tau_true"].to_numpy()
        true_train = float(indexed.loc[train.unit_id, "tau_true"].mean())
        assumption = {
            "A": "Randomized design.",
            "C": "Assume exchangeability given observed engagement and loyalty, consistency and positivity.",
            "F": "Hidden confounding remains; estimates are not identified causal effects.",
        }[scenario]
        pair = preds["logistic"]
        rows, m, bal, sens, diag = _estimates(
            scenario, x, t, y, pv, pair.mu0, pair.mu1, tau, identification=assumption
        )
        dr = aipw(tt, ty, pe, a, b)
        ate.append(
            _result(
                scenario,
                "aipw_crossfit",
                "ATE",
                "training OOF",
                len(train),
                dr.pop("estimate"),
                true_train,
                warning=assumption + " Approximate independent-row interval.",
                **dr,
            )
        )
        other = preds["lightgbm"]
        ate.append(
            _result(
                scenario,
                "regression_lightgbm",
                "ATE",
                "validation",
                len(val),
                standardize(other.mu0, other.mu1),
                float(tau.mean()),
                warning=assumption,
            )
        )
        if scenario == "C":
            true_e = indexed.loc[val.unit_id, "propensity_true"].to_numpy()
            sensitivity.append(
                {
                    "dataset": "C",
                    "profile": "oracle propensity",
                    "target": "validation ATE",
                    "n": len(val),
                    "excluded": 0,
                    "truth": float(tau.mean()),
                    "hajek": weighted_contrast(t, y, weights(t, true_e)),
                    "aipw": aipw(t, y, true_e, pair.mu0, pair.mu1)["estimate"],
                }
            )
        ate.extend(row for row in rows if row["estimand"].startswith("ATE"))
        att.extend(row for row in rows if row["estimand"].endswith("ATT"))
        matching.append(m)
        balance.extend(bal)
        sensitivity.extend(sens)
        diagnostics.extend(diag)
        provenance[scenario] = {
            "propensity": source,
            "outcome": "arm logistic; tree standardization comparator",
            "validation_n": len(val),
            "training_oof_n": len(train),
            "fold_sizes": np.bincount(outcome_folds).tolist(),
            "validation_fingerprint": training_fingerprint(
                val, spec, spec.treatment_column
            ),
        }
    external, ms, bs, ds, nsw = (
        ([], [], [], [], {"status": "offline fixture"}) if fixture else _nsw(root, seed)
    )
    robust = robustness(seed, 8 if fixture else 80, n=400 if fixture else 1800)
    for name, rows in (
        ("observational_ate_results", ate),
        ("observational_att_results", att),
        ("matching_diagnostics", matching + ms),
        ("weight_diagnostics", diagnostics + ds),
        ("overlap_sensitivity", sensitivity),
        ("covariate_balance_comparison", balance + bs),
        ("doubly_robust_simulation", robust),
        ("lalonde_benchmark", external),
    ):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    _plots(target, balance, ate)
    stats = {
        name: {
            "mean": float(g.estimate.mean()),
            "bias": float(g.error.mean()),
            "rmse": float(np.sqrt(np.mean(g.error**2))),
            "coverage": float(g.covers.mean()),
            "repetitions": len(g),
        }
        for name, g in pd.DataFrame(robust).groupby("specification")
    }
    summary = {
        "runtime_seconds": perf_counter() - start,
        "seed": seed,
        "fixture": fixture,
        "nsw": nsw,
        "provenance": provenance,
        "robustness": stats,
    }
    write_json(target / "session_09_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download:
        fetch_nsw(args.root)
    print(run(args.root, fixture=args.fixture, output=args.output, seed=args.seed))


if __name__ == "__main__":
    main()
