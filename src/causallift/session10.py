"""Refutation and sensitivity diagnostics on fixed development partitions."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LinearRegression

from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.splits import split_data
from causallift.meta_learners import (
    MetaLearner,
    fit_binary_outcome,
    outcome_configuration,
    outcome_probability,
)
from causallift.observational import (
    aipw,
    effective_n,
    match,
    overlap_profiles,
    standardize,
    weighted_contrast,
    weighted_smd,
    weights,
)
from causallift.prediction import training_fingerprint
from causallift.session01 import write_json
from causallift.session09 import _nsw, load_saved_propensity
from causallift.simulation import simulate

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

STRENGTHS = (0.0, 0.4, 0.8, 1.2, 1.6)


def shuffle_assignments(t, rng):
    return rng.permutation(t)


def subset_positions(t, rng, fraction=0.8):
    if not 0 < fraction <= 1:
        raise ValueError("Subset fraction must be in (0, 1]")
    return np.sort(
        np.concatenate(
            [
                rng.choice(p, max(1, int(len(p) * fraction)), replace=False)
                for arm in (0, 1)
                if len(p := np.flatnonzero(t == arm))
            ]
        )
    )


def add_noise(x, xv, rng):
    return x.assign(__independent_noise=rng.normal(size=len(x))), xv.assign(
        __independent_noise=rng.normal(size=len(xv))
    )


def position_digest(units):
    return hashlib.sha256(np.asarray(units, dtype="<i8").tobytes()).hexdigest()


def estimate(x, t, y, xv, tv, yv, *, seed, binary=True, design=None):
    t = np.asarray(t, dtype=bool)
    if (
        y.shape != t.shape
        or len(x) != len(t)
        or len(xv) != len(tv)
        or set(t) != {False, True}
    ):
        raise ValueError("Aligned treatment arms are required")
    cfg = outcome_configuration("logistic", seed, 55)
    if design is None:
        p = outcome_probability(fit_binary_outcome(x, t, cfg, tuple(x.columns)), xv)
    else:
        p = np.full(len(xv), design)
    if binary:
        potential = (
            MetaLearner("t", "logistic", seed=seed)
            .fit(x, t, y)
            .predict_potential_outcomes(xv)
        )
        a, b = potential.mu0, potential.mu1
    else:
        a, b = [
            LinearRegression().fit(x.loc[mask], y[mask]).predict(xv) for mask in (~t, t)
        ]
    return {
        "naive": weighted_contrast(tv, yv, np.ones(len(tv))),
        "regression": standardize(a, b),
        "hajek": weighted_contrast(tv, yv, weights(tv, p)),
        "aipw": aipw(tv, yv, p, a, b),
        "propensity": p,
    }


def continuous_design(rng, n, strength):
    x, u, error, nc_error = rng.normal(size=(4, n))
    score = expit(0.7 * x + strength * u)
    t = rng.binomial(1, score)
    y = 1.0 * t + 0.9 * x + strength * u + error
    nc = 0.9 * x + strength * u + nc_error
    return pd.DataFrame({"x": x, "u": u}), t, y, nc, score


def continuous_checks(seed, repeats, fixture):
    rows, negative = [], []
    for idx, strength in enumerate(STRENGTHS):
        for rep in range(repeats):
            sub = int(np.random.SeedSequence([seed, rep, 702]).generate_state(1)[0])
            x, t, y, nc, p = continuous_design(
                np.random.default_rng(sub), 500 if fixture else 2400, strength
            )
            split = len(x) // 2
            for info in ("observed", "full_information"):
                keep = ["x"] if info == "observed" else ["x", "u"]
                a, b = x[keep].iloc[:split], x[keep].iloc[split:]
                output = estimate(
                    a,
                    t[:split],
                    y[:split],
                    b,
                    t[split:],
                    y[split:],
                    seed=sub,
                    binary=False,
                )
                q = output["propensity"]
                for method in ("naive", "regression", "hajek", "aipw"):
                    if info == "full_information" and method != "aipw":
                        continue
                    dr = output["aipw"] if method == "aipw" else {}
                    point = dr["estimate"] if dr else output[method]
                    rows.append(
                        {
                            "strength": strength,
                            "rep": rep,
                            "seed": sub,
                            "method": method,
                            "information": info,
                            "estimate": point,
                            "truth": 1.0,
                            "error": point - 1,
                            "n": len(q),
                            "true_extreme_share": float(
                                np.mean((p[split:] < 0.05) | (p[split:] > 0.95))
                            ),
                            "fitted_extreme_share": float(
                                np.mean((q < 0.05) | (q > 0.95))
                            ),
                            "ess": effective_n(weights(t[split:], q)),
                            "covers": dr["ci_low"] <= 1 <= dr["ci_high"]
                            if dr
                            else None,
                        }
                    )
            if rep == 0 and idx in (0, len(STRENGTHS) - 1):
                for outcome, observed, truth in (
                    ("main", y, 1.0),
                    ("negative_control", nc, 0.0),
                ):
                    a, b = x[["x"]].iloc[:split], x[["x"]].iloc[split:]
                    r = estimate(
                        a,
                        t[:split],
                        observed[:split],
                        b,
                        t[split:],
                        observed[split:],
                        seed=sub,
                        binary=False,
                    )
                    negative.append(
                        {
                            "hidden_strength": strength,
                            "seed": sub,
                            "outcome": outcome,
                            "truth": truth,
                            "n": len(b),
                            "naive": r["naive"],
                            "regression": r["regression"],
                            **r["aipw"],
                        }
                    )
    return rows, negative


def _inputs(root, fixture, seed):
    for scenario in "ACF":
        if fixture:
            frame, truth, spec = simulate(scenario, 800, seed)
            frame["split"] = split_data(frame, spec, seed)
            train, val = (frame.loc[frame.split.eq(s)] for s in ("train", "validation"))
        else:
            path = root / f"data/synthetic/synthetic_{scenario}.parquet"
            spec = DatasetSpec(
                **json.loads(path.with_suffix(".spec.json").read_text(encoding="utf-8"))
            )
            train, val = (load_dataset(path, spec, s) for s in ("train", "validation"))
            truth = pd.read_parquet(
                path.with_stem(path.stem + "_truth"), columns=["unit_id", "tau_true"]
            )
        yield scenario, train, val, spec, truth


def _matching(label, t, y, e, tau, legacy):
    ti, ci, d, r = match(t, e, legacy_ties=legacy)
    reuse = np.bincount(ci)
    return {
        "dataset": label,
        "version": "session09_legacy" if legacy else "corrected",
        "matched_treated": len(ti),
        "unique_controls": int(np.count_nonzero(reuse)),
        "max_control_reuse": int(reuse.max()) if len(ci) else 0,
        "control_ess": effective_n(reuse) if len(ci) else None,
        "mean_distance": float(d.mean()) if len(ci) else None,
        "max_distance": float(d.max()) if len(ci) else None,
        "logit_caliper": r,
        "estimate": float((y[ti] - y[ci]).mean()) if len(ci) else None,
        "truth": float(tau[ti].mean()) if len(ti) else None,
        "inference": "unsupported",
    }


def _sensitivity(label, xv, t, y, tau, probs, a, b):
    rows = []
    for name, p in probs.items():
        for profile, (keep, prob) in overlap_profiles(p).items():
            tv = t[keep].astype(bool)
            if not tv.any() or tv.all():
                rows.append(
                    {
                        "dataset": label,
                        "model": name,
                        "profile": profile,
                        "n": int(keep.sum()),
                        "excluded": int((~keep).sum()),
                        "status": "insufficient_support",
                    }
                )
                continue
            w = weights(tv, prob[keep])
            balance = {}
            for feature in xv:
                z = xv[feature].to_numpy()
                denominator = np.sqrt(
                    (z[t == 1].var(ddof=1) + z[t == 0].var(ddof=1)) / 2
                )
                balance[feature] = weighted_smd(tv, z[keep], w, denominator=denominator)
            rows.append(
                {
                    "dataset": label,
                    "model": name,
                    "profile": profile,
                    "n": int(keep.sum()),
                    "estimand": "overlap-restricted validation ATE"
                    if not keep.all()
                    else "validation ATE",
                    "excluded": int((~keep).sum()),
                    "clipped_scores": int(np.sum(prob != p)),
                    "true_ate": float(tau[keep].mean()),
                    "maximum_weight": float(w.max()),
                    "ess_treated": effective_n(w[tv]),
                    "ess_control": effective_n(w[~tv]),
                    "ess_treated_fraction": effective_n(w[tv]) / int(tv.sum()),
                    "ess_control_fraction": effective_n(w[~tv]) / int((~tv).sum()),
                    "weighted_smd": json.dumps(balance),
                    "hajek": weighted_contrast(tv, y[keep], w),
                    "aipw": aipw(tv, y[keep], prob[keep], a[keep], b[keep])["estimate"],
                }
            )
    return rows


def _describe(values):
    data = np.asarray(values)
    return {
        "mean": float(data.mean()),
        "sd": float(data.std(ddof=1)) if len(data) > 1 else None,
        "n": len(data),
        "q05": float(np.quantile(data, 0.05)),
        "q95": float(np.quantile(data, 0.95)),
    }


def reliability_status(identified, overlap_warning):
    if identified is False:
        return "not_identified"
    if identified is None:
        return "not_assessed"
    if overlap_warning is None:
        return "not_assessed"
    return "caution" if overlap_warning else "supported_under_assumptions"


def run(
    root,
    *,
    fixture=False,
    output=None,
    seed=42,
    placebo_reps=40,
    noise_reps=15,
    subset_reps=20,
    hidden_reps=24,
):
    start = perf_counter()
    if min(placebo_reps, noise_reps, subset_reps, hidden_reps) < 1:
        raise ValueError("Repetition counts must be positive")
    if fixture:
        placebo_reps, noise_reps, subset_reps, hidden_reps = [
            min(n, 3) for n in (placebo_reps, noise_reps, subset_reps, hidden_reps)
        ]
    target = output or root / "artifacts" / (
        "session_10_fixture" if fixture else "session_10"
    )
    target.mkdir(parents=True, exist_ok=True)
    matching, placebo, noise, subsets, sensitivity = [], [], [], [], []
    reliability = {}
    rng = np.random.default_rng(seed)
    folder = root / "artifacts/session_08"
    for scenario, train, val, spec, truth in _inputs(root, fixture, seed):
        x, xv = spec.features(train), spec.features(val)
        t, y = train.treatment.to_numpy(), train.outcome.to_numpy()
        tv, yv = val.treatment.to_numpy(), val.outcome.to_numpy()
        design = spec.assignment_probability if scenario == "A" else None
        probs = {}
        for method in ("logistic", "lightgbm"):
            if fixture:
                cfg = outcome_configuration(method, seed, 55)
                p = outcome_probability(
                    fit_binary_outcome(x, t, cfg, tuple(x.columns)), xv
                )
            else:
                _, p, _ = load_saved_propensity(
                    folder, scenario, spec, train, val, 42, model=method
                )
            probs[method] = p
        if scenario == "A":
            probs["known_design"] = np.full(len(val), design)
        prop = probs["known_design" if scenario == "A" else "logistic"]
        tau = truth.set_index("unit_id").loc[val.unit_id, "tau_true"].to_numpy()
        potential = (
            MetaLearner("t", "logistic", seed=seed)
            .fit(x, t, y)
            .predict_potential_outcomes(xv)
        )
        base = aipw(tv, yv, prop, potential.mu0, potential.mu1)["estimate"]
        matching.extend(
            _matching(scenario, tv, yv, prop, tau, legacy) for legacy in (True, False)
        )
        s = _sensitivity(scenario, xv, tv, yv, tau, probs, potential.mu0, potential.mu1)
        sensitivity.extend(s)
        if scenario == "A":
            for rep in range(placebo_reps):
                child = int(rng.integers(2**31))
                stream = np.random.default_rng(child)
                pt, pv = shuffle_assignments(t, stream), shuffle_assignments(tv, stream)
                r = estimate(x, pt, y, xv, pv, yv, seed=child, design=design)["aipw"]
                placebo.append(
                    {
                        "dataset": scenario,
                        "rep": rep,
                        "seed": child,
                        "original": base,
                        "n": len(tv),
                        "treated_train": int(pt.sum()),
                        "treated_validation": int(pv.sum()),
                        **r,
                        "rejects_zero": r["ci_low"] > 0 or r["ci_high"] < 0,
                    }
                )
        if scenario == "C":
            for rep in range(noise_reps):
                child = int(rng.integers(2**31))
                a, b = add_noise(x, xv, np.random.default_rng(child))
                r = estimate(a, t, y, b, tv, yv, seed=child)["aipw"]
                noise.append(
                    {
                        "dataset": scenario,
                        "rep": rep,
                        "seed": child,
                        "n": len(tv),
                        "original": base,
                        "change": r["estimate"] - base,
                        "absolute_change": abs(r["estimate"] - base),
                        **r,
                    }
                )
        for rep in range(subset_reps):
            child = int(rng.integers(2**31))
            stream = np.random.default_rng(child)
            a, b = subset_positions(t, stream), subset_positions(tv, stream)
            r = estimate(
                x.iloc[a],
                t[a],
                y[a],
                xv.iloc[b],
                tv[b],
                yv[b],
                seed=child,
                design=design,
            )["aipw"]
            actual = float(tau[b].mean())
            subsets.append(
                {
                    "dataset": scenario,
                    "rep": rep,
                    "seed": child,
                    "n": len(b),
                    "train_n": len(a),
                    "target": "80% stratified validation subset ATE",
                    "training_units_sha256": position_digest(train.unit_id.iloc[a]),
                    "validation_units_sha256": position_digest(val.unit_id.iloc[b]),
                    "original": base,
                    "truth": actual,
                    "error": r["estimate"] - actual,
                    **r,
                }
            )
        unmodified = next(
            r
            for r in s
            if r["profile"] == "unmodified"
            and r["model"] == ("known_design" if scenario == "A" else "logistic")
        )
        warning = (
            unmodified["maximum_weight"] > 20
            or min(
                unmodified[k] for k in ("ess_treated_fraction", "ess_control_fraction")
            )
            < 0.5
        )
        reliability[scenario] = {
            "estimand": "validation ATE",
            "status": reliability_status(scenario != "F", warning),
            "identification": "randomized"
            if scenario == "A"
            else "observed backdoor assumed"
            if scenario == "C"
            else "hidden backdoor open",
            "adjustment_set": []
            if scenario == "A"
            else ["engagement", "loyalty"]
            if scenario == "C"
            else None,
            "fitted_features": list(spec.feature_columns),
            "assumptions": [
                "consistency",
                "positivity",
                "exchangeability under stated design",
            ],
            "overlap_status": "weight concentration" if warning else "no weight flag",
            "weights_and_balance": unmodified,
            "extreme_weight_warning": warning,
            "original_aipw": base,
            "subset_stability": {
                field: _describe(
                    [r[field] for r in subsets if r["dataset"] == scenario]
                )
                for field in ("estimate", "truth", "error")
            },
            "placebo": None
            if scenario != "A"
            else _describe([r["estimate"] for r in placebo])
            | {
                "nominal_rejection_share": float(
                    np.mean([r["rejects_zero"] for r in placebo])
                )
            },
            "random_feature": None
            if scenario != "C"
            else _describe([r["change"] for r in noise]),
            "negative_control": "separate simulation only",
            "uncertainty": "Approximate Wald intervals; subset quantiles are descriptive; matching inference unsupported.",
            "hidden_confounding": "A successful refutation cannot establish identification; the strength sweep is not a real-world bound.",
            "train_fingerprint": training_fingerprint(train, spec, "treatment"),
            "validation_fingerprint": training_fingerprint(val, spec, "treatment"),
            "train_outcome_fingerprint": training_fingerprint(train, spec, "outcome"),
            "validation_outcome_fingerprint": training_fingerprint(
                val, spec, "outcome"
            ),
        }
    if not fixture:
        for legacy in (True, False):
            _, diag, _, _, _ = _nsw(root, 42, legacy_ties=legacy)
            matching.extend(
                r
                | {
                    "version": "session09_legacy" if legacy else "corrected",
                    "truth": None,
                }
                for r in diag
            )
    hidden, negative = continuous_checks(seed, hidden_reps, fixture)
    summaries = []
    for (strength, method, info), g in pd.DataFrame(hidden).groupby(
        ["strength", "method", "information"]
    ):
        summaries.append(
            {
                "strength": strength,
                "method": method,
                "information": info,
                "n_reps": len(g),
                "truth": 1.0,
                "mean_estimate": float(g.estimate.mean()),
                "bias": float(g.error.mean()),
                "rmse": float(np.sqrt(np.mean(g.error**2))),
                "mean_ess": float(g.ess.mean()),
                "true_extreme_share": float(g.true_extreme_share.mean()),
                "fitted_extreme_share": float(g.fitted_extreme_share.mean()),
                "coverage": None
                if method != "aipw"
                else float(g.covers.astype(float).mean()),
            }
        )
    for name, rows in (
        ("matching_tie_correction", matching),
        ("placebo_refutation", placebo),
        ("random_common_cause", noise),
        ("subset_stability", subsets),
        ("negative_control_results", negative),
        ("hidden_confounding_sensitivity", summaries),
        ("overlap_sensitivity", sensitivity),
    ):
        pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), layout="constrained")
    for ax, label, rows in zip(
        axs,
        ("A placebo", "C noise feature", "C subset"),
        (placebo, noise, [r for r in subsets if r["dataset"] == "C"]),
        strict=True,
    ):
        ax.hist([r["estimate"] * 100 for r in rows], bins=10, color="#28618b")
        ax.axvline(
            0 if label == "A placebo" else rows[0]["original"] * 100,
            color="#a54a2a",
            linestyle="--",
        )
        ax.set(title=label, xlabel="AIPW (pp)")
    fig.savefig(target / "refutation_summary.png", dpi=145)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for (method, info), g in pd.DataFrame(summaries).groupby(["method", "information"]):
        ax.plot(g.strength, g.mean_estimate, marker="o", label=f"{method} · {info}")
    ax.axhline(1, color="gray", linestyle="--")
    ax.set(xlabel="Hidden confounding strength", ylabel="Mean estimated effect")
    ax.legend(fontsize=8)
    fig.savefig(target / "sensitivity_analysis.png", dpi=145)
    plt.close(fig)
    write_json(
        target / "causal_reliability_report.json",
        {"datasets": reliability, "session07_dml_coverage": "0/30 preserved"},
    )
    summary = {
        "runtime_seconds": perf_counter() - start,
        "seed": seed,
        "fixture": fixture,
        "placebo_reps": placebo_reps,
        "noise_reps": noise_reps,
        "subset_reps": subset_reps,
        "hidden_reps": hidden_reps,
        "placebo": reliability["A"]["placebo"],
        "random_feature": reliability["C"]["random_feature"],
        "nsw_available": any(r["dataset"] not in "ACF" for r in matching),
    }
    write_json(target / "session_10_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output", type=Path)
    p.add_argument("--fixture", action="store_true")
    for name, n in (
        ("seed", 42),
        ("placebo-reps", 40),
        ("noise-reps", 15),
        ("subset-reps", 20),
        ("hidden-reps", 24),
    ):
        p.add_argument("--" + name, type=int, default=n)
    print(run(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
