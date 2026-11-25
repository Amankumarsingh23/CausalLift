"""Replay the frozen policy evaluation with a persisted pre-outcome action lock."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from causallift.data.contract import DatasetSpec, load_dataset
from causallift.data.criteo import CRITEO, checksum
from causallift.data.splits import PARTITIONS, split_data
from causallift.final_reporting import plots
from causallift.optimization import allocate
from causallift.platform.artifact_registry import FROZEN_HASHES, PROTOCOL
from causallift.policy_evaluation import INTERVAL_NOTE, PolicyEvaluation
from causallift.prediction import training_fingerprint
from causallift.session01 import write_json
from causallift.session06 import _manifest
from causallift.session07 import _fingerprint
from causallift.session12 import _action_hash, _record
from causallift.simulation import simulate
from causallift.targeting import aligned_synthetic_effects

POLICIES = ("none", "random", "predictive", "uplift", "profit")
CRITEO_ROOTS = {
    "Criteo 200k": "",
    "Criteo 1M": "data/processed/benchmark_1m",
}


def now():
    return datetime.now(UTC).isoformat()


def verify_protocol(root):
    path = root / PROTOCOL
    digest = checksum(path)
    if digest not in FROZEN_HASHES:
        raise ValueError(f"Frozen protocol hash mismatch: {digest}")
    return json.loads(path.read_text(encoding="utf-8")), digest


def load_frozen(root, protocol, entries, dataset, spec, train):
    outcome = spec.outcome_column
    predictive, causal = entries
    variants = (
        (
            "04",
            predictive[dataset, outcome, "lightgbm"],
            training_fingerprint(train, spec, outcome),
        ),
        (
            "05",
            causal[dataset, outcome, "s_lightgbm"],
            _fingerprint(train, spec, outcome),
        ),
        (
            "05",
            causal[dataset, outcome, "t_logistic"],
            _fingerprint(train, spec, outcome),
        ),
    )
    paths, declared = [], {}
    for session, entry, training in variants:
        name = f"session_{session}/{entry['model_path']}"
        expected = protocol["model_artifact_sha256"][name]
        path = root / "artifacts" / name
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing frozen model {name}; expected SHA-256 {expected}"
            )
        actual = checksum(path)
        if actual != expected:
            raise ValueError(f"Frozen model {name}: expected {expected}, got {actual}")
        if entry["training_fingerprint_sha256"] != training or tuple(
            entry["feature_allowlist"]
        ) != tuple(spec.feature_columns):
            raise ValueError(
                f"Training fingerprint or feature allowlist mismatch: {name}"
            )
        declared[name] = {
            "sha256": actual,
            "training_fingerprint": training,
            "features": list(spec.feature_columns),
        }
        paths.append(path)
    return [joblib.load(path) for path in paths], declared


def signature(path):
    info = path.stat()
    return info.st_size, info.st_mtime_ns


def stage_projection(path, spec, split_counts):
    columns = ["split", *spec.feature_columns]
    if spec.unit_id:
        columns.append(spec.unit_id)
    view = pd.read_parquet(path, columns=columns)
    counts = {str(key): int(size) for key, size in view.groupby("split").size().items()}
    if set(counts) != set(PARTITIONS) or counts != split_counts:
        raise ValueError(
            f"Persisted split counts differ for {spec.dataset_id}: {counts}"
        )
    ids = view[spec.unit_id].to_numpy() if spec.unit_id else np.arange(len(view))
    if pd.isna(ids).any() or not pd.Index(ids).is_unique:
        raise ValueError(f"Invalid or shared row identities in {spec.dataset_id}")
    projected = pd.util.hash_pandas_object(view, index=False).to_numpy(dtype="<u8")
    split_hashes = {
        split: _action_hash(ids, view.split.eq(split).to_numpy())
        for split in PARTITIONS
    }
    mask = view.split.eq("test")
    x = spec.features(view.loc[mask]).reset_index(drop=True)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("Test feature projection is not finite")
    return (
        x,
        ids[mask],
        {
            "split_counts": counts,
            "split_identity_sha256": split_hashes,
            "safe_dataset_projection_sha256": hashlib.sha256(
                projected.tobytes()
            ).hexdigest(),
        },
    )


def economics(protocol, dataset, n):
    seed = protocol["seed"]
    synthetic = dataset.startswith("synthetic_")
    if synthetic:
        cfg = protocol["synthetic_economics"]
        first, _, last = cfg["eligibility_seed_components"]
        rng = np.random.default_rng(
            np.random.SeedSequence([first, ord(dataset[-1]), last])
        )
        eligible = rng.random(n) >= round(1 - cfg["eligible_probability"], 12)
        value = np.full(n, cfg["contribution"])
        cost = np.full(n, cfg["cost"])
        budgets = [
            (f, int(cfg["cost"] * n * f), int(cfg["capacity_fraction"] * n))
            for f in cfg["fractions"]
        ]
    else:
        cfg = protocol["criteo_economics"]
        if (
            cfg["contribution"] != "synthetic integer draws 60–140 inclusive"
            or cfg["eligibility"] != "all"
        ):
            raise ValueError("Unknown frozen Criteo economic rule")
        value = np.random.default_rng(seed).integers(60, 141, n)
        cost = np.full(n, cfg["cost"])
        eligible = np.ones(n, dtype=bool)
        budgets = [
            (f, round(cfg["cost"] * int(f * n), 2), int(f * n))
            for f in cfg["fractions"]
        ]
    return value, cost, eligible, budgets


@dataclass
class LockedCohort:
    name: str
    path: Path
    spec: DatasetSpec
    ids: np.ndarray
    value: np.ndarray
    cost: np.ndarray
    mu0: np.ndarray
    mu1: np.ndarray
    lockfile: Path
    metadata: dict


def stage(name, path, spec, x, ids, scores, target, protocol, metadata):
    effect, prediction, mu0, mu1 = scores
    n = len(ids)
    value, cost, eligible, budgets = economics(protocol, name, n)
    actions, hashes = {}, []
    for fraction, budget, capacity in budgets:
        for policy, score in zip(
            POLICIES, (None, np.zeros(n), prediction, effect, None), strict=True
        ):
            chosen, diagnostics = allocate(
                effect * value - cost,
                cost,
                eligible,
                budget,
                capacity=capacity,
                policy={"none": "none", "profit": "uniform"}.get(policy, "ranking"),
                score=score,
                seed=protocol["seed"],
            )
            action = np.zeros(n, dtype=bool)
            action[chosen] = True
            key = f"{fraction:.2f}_{policy}"
            actions[key] = action
            hashes.append(
                dict(
                    key=key,
                    fraction=fraction,
                    budget=budget,
                    capacity=capacity,
                    policy=policy,
                    selected=len(chosen),
                    spend=diagnostics["spend"],
                    sha256=_action_hash(ids, action),
                )
            )
    local = target / f"local_actions_{name.lower().replace(' ', '_')}.parquet"
    pd.DataFrame({"unit_id": ids, **actions}).to_parquet(local, index=False)
    metadata.update(
        {
            "action_vectors": hashes,
            "action_archive_sha256": checksum(local),
            "file_signature_at_lock": signature(path),
            "test_rows": n,
            "test_feature_columns": list(x.columns),
        }
    )
    return LockedCohort(name, path, spec, ids, value, cost, mu0, mu1, local, metadata)


def evaluate_locked(cohort, frozen, results, support, paired, designation):
    if checksum(cohort.lockfile) != frozen["action_archive_sha256"]:
        raise ValueError("Action archive changed after the lock")
    actions = pd.read_parquet(cohort.lockfile)
    if not np.array_equal(actions.unit_id, cohort.ids):
        raise ValueError("Action rows changed after the lock")
    for record in frozen["action_vectors"]:
        if _action_hash(cohort.ids, actions[record["key"]]) != record["sha256"]:
            raise ValueError("Action vector changed after the lock")
    if list(signature(cohort.path)) != frozen["file_signature_at_lock"]:
        raise ValueError("Observed data file changed after the lock")
    spec = cohort.spec
    columns = [spec.treatment_column, spec.outcome_column]
    if spec.unit_id:
        columns.append(spec.unit_id)
    observed = pd.read_parquet(
        cohort.path, columns=columns, filters=[("split", "=", "test")]
    )
    if len(observed) != len(cohort.ids) or (
        spec.unit_id and not np.array_equal(observed[spec.unit_id], cohort.ids)
    ):
        raise ValueError("Outcome rows do not align with locked actions")
    t, y = observed[spec.treatment_column], observed[spec.outcome_column]
    synthetic = spec.unit_id is not None
    truth = (
        aligned_synthetic_effects(
            observed, spec, cohort.path.with_stem(cohort.path.stem + "_truth")
        )
        if synthetic
        else None
    )
    propensity = spec.assignment_probability if synthetic else float(t.mean())
    evaluator = PolicyEvaluation(
        cohort.ids,
        t,
        y,
        propensity,
        cohort.mu0,
        cohort.mu1,
        contribution=cohort.value,
        cost=cohort.cost,
    )
    frozen.update(
        {
            "outcome_accessed_utc": now(),
            "observed_file_sha256": checksum(cohort.path),
            "events": int(y.sum()),
            "control_events": int(y[t.eq(0)].sum()),
            "treated": int(t.sum()),
        }
    )
    dataset = cohort.name[-1] if synthetic else cohort.name
    for fraction in sorted({row["fraction"] for row in frozen["action_vectors"]}):
        records = {
            row["policy"]: row
            for row in frozen["action_vectors"]
            if row["fraction"] == fraction
        }
        subset = {
            label: actions[row["key"]].to_numpy() for label, row in records.items()
        }
        keys = {
            "dataset": dataset,
            "split": designation,
            "fraction": fraction,
            "economy": "uniform" if synthetic else "hypothetical",
            "outcome": spec.outcome_column,
            "budget": records["profit"]["budget"],
            "capacity": records["profit"]["capacity"],
            "positives": int(y.sum()),
            "control_positives": int(y[t.eq(0)].sum()),
            "logging_propensity": propensity,
            "identification": "supported_under_assumptions"
            if synthetic
            else "not_assessed",
            "logging_propensity_source": "known randomized design"
            if synthetic
            else "released test assignment share",
        }
        start = len(paired)
        _record(results, support, paired, evaluator, subset, keys, truth, True)
        for reference in ("predictive", "none"):
            if reference == "none":
                paired.extend(
                    keys
                    | {
                        "policy": "profit",
                        "reference": reference,
                        "allocation_hash": records["profit"]["sha256"],
                        "reference_allocation_hash": records["none"]["sha256"],
                    }
                    | row
                    for row in evaluator.compare(subset["profit"], subset["none"])
                )
            target = subset["profit"].astype(int) - subset[reference].astype(int)
            for row in paired[start:]:
                if row["reference"] == reference:
                    row.update(
                        {
                            "reference_selected": records[reference]["selected"],
                            "reference_ess": evaluator.support(
                                subset[reference],
                                evaluator.scores(subset[reference])[1],
                            )["ess"],
                            "true_conversions_difference": None
                            if truth is None
                            else float(truth @ target),
                            "true_profit_difference": None
                            if truth is None
                            else float((truth * cohort.value - cohort.cost) @ target),
                        }
                    )


def validation_comparison(root, paired):
    historical = pd.read_csv(root / "artifacts/session_01/split_summary.csv")
    historical = historical.loc[historical.split.eq("validation")]
    val = pd.read_csv(root / "artifacts/session_12/paired_policy_comparisons.csv")
    synthetic = pd.read_csv(
        root / "artifacts/session_12/synthetic_policy_evaluation.csv"
    )
    criteo = pd.read_csv(root / "artifacts/session_12/criteo_policy_evaluation.csv")
    rows = []
    for final in paired:
        if final["method"] != "dr" or final["fraction"] != 0.20:
            continue
        dataset, reference = final["dataset"], final["reference"]
        table = (
            val
            if reference == "predictive"
            else (criteo if dataset.startswith("Criteo") else synthetic)
        )
        budget = {"Criteo 1M": 16000, "Criteo 200k": 3200}.get(dataset, 1600)
        match = table.loc[
            table.dataset.eq(dataset)
            & table.method.eq("dr")
            & table.policy.eq("profit")
            & table.economy.eq(final["economy"])
            & table.budget.eq(budget)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Validation comparison missing for {dataset} / {reference}"
            )
        previous = match.iloc[0]
        entry = {
            "dataset": dataset,
            "reference": reference,
            "fraction": 0.20,
            "validation_budget": budget,
            "test_budget": final["budget"],
        }
        for field in (
            "n",
            "incremental_profit",
            "incremental_conversions",
            "ci_profit_low",
            "ci_profit_high",
            "spend",
            "reference_spend",
            "positives",
            "control_positives",
        ):
            entry[f"validation_{field}"] = previous.get(field)
            entry[f"test_{field}"] = final[field]
        if final["economy"] == "uniform":
            arms = historical.loc[historical.dataset.eq(f"synthetic_{dataset}")]
            events = (arms.rows * arms.outcome_rate).round().astype(int)
            entry["validation_positives"] = int(events.sum())
            entry["validation_control_positives"] = int(
                events[arms.treatment.eq(0)].sum()
            )
        for split, row in (("validation", previous), ("test", final)):
            entry[f"{split}_profit_per_unit"] = row["incremental_profit"] / row["n"]
            entry[f"{split}_conversions_per_1000"] = (
                1000 * row["incremental_conversions"] / row["n"]
            )
            entry[f"{split}_direction"] = (
                "positive"
                if row["incremental_profit"] > 0
                else "negative"
                if row["incremental_profit"] < 0
                else "zero"
            )
        entry["test_minus_validation_profit_per_unit"] = (
            entry["test_profit_per_unit"] - entry["validation_profit_per_unit"]
        )
        entry["absolute_profit_per_unit_change"] = abs(
            entry["test_minus_validation_profit_per_unit"]
        )
        rows.append(entry)
    return rows


def run(root, target, fixture=False):
    start = perf_counter()
    protocol, digest = verify_protocol(root)
    designation = "fixture" if fixture else "reserved_test"
    target.mkdir(parents=True, exist_ok=True)
    lock = target / "action_freeze.json"
    if lock.exists():
        raise FileExistsError(
            f"Preserve the previous action freeze; choose another output: {target}"
        )
    entries = (
        None
        if fixture
        else (
            _manifest(
                root / "artifacts/session_04/predictive_model_manifest.json",
                "model",
                "lightgbm",
            ),
            _manifest(root / "artifacts/session_05/causal_model_manifest.json"),
        )
    )
    names = (
        ["synthetic_E"]
        if fixture
        else protocol["primary_cohorts"] + protocol["secondary_cohorts"]
    )
    cohorts, blocked = [], {}
    for name in names:
        try:
            if fixture:
                sample, truth, spec = simulate("E", 800, protocol["seed"])
                sample["split"] = split_data(sample, spec, protocol["seed"])
                path = target / "local_fixture.parquet"
                sample.to_parquet(path, index=False)
                truth.to_parquet(path.with_stem(path.stem + "_truth"), index=False)
                counts = sample.split.value_counts().to_dict()
            else:
                base = root / CRITEO_ROOTS.get(name, "")
                path = base / (
                    f"data/synthetic/{name}.parquet"
                    if name.startswith("synthetic_")
                    else "data/processed/criteo_v2_1.parquet"
                )
                spec = (
                    CRITEO
                    if name in CRITEO_ROOTS
                    else DatasetSpec(
                        **json.loads(
                            path.with_suffix(".spec.json").read_text(encoding="utf-8")
                        )
                    )
                )
                published = json.loads(
                    (base / "artifacts/session_01/dataset_summary.json").read_text(
                        encoding="utf-8"
                    )
                )
                counts = published["datasets"][spec.dataset_id]["split_sizes"]
            initial = signature(path)
            x, ids, metadata = stage_projection(path, spec, counts)
            if fixture:
                # Deterministic fixture scores exercise the mechanism without loading models.
                feature = np.tanh(x.engagement.to_numpy())
                scores = (
                    0.05 + 0.09 * feature,
                    0.35 + 0.08 * feature,
                    np.full(len(x), 0.30),
                    0.35 + 0.09 * feature,
                )
            else:
                train = load_dataset(path, spec, "train")
                if len(train) != counts["train"] or not train.split.eq("train").all():
                    raise ValueError("Training partition differs from persisted counts")
                if spec.unit_id and (
                    not train[spec.unit_id].is_unique
                    or pd.Index(ids).isin(train[spec.unit_id]).any()
                ):
                    raise ValueError("Training and test identities overlap")
                (predictive, causal, nuisance), metadata["verified_models"] = (
                    load_frozen(root, protocol, entries, name, spec, train)
                )
                potential = nuisance.predict_potential_outcomes(x)
                scores = (
                    causal.predict_effect(x),
                    predictive.predict(x),
                    potential.mu0,
                    potential.mu1,
                )
            if initial != signature(path):
                raise ValueError("Observed data changed during feature preparation")
            cohorts.append(
                stage(name, path, spec, x, ids, scores, target, protocol, metadata)
            )
        except (ValueError, KeyError, OSError) as exc:
            blocked[name] = str(exc)
    write_json(
        lock,
        {
            "locked_utc": now(),
            "split": designation,
            "protocol_sha256": digest,
            "cohorts": {cohort.name: cohort.metadata for cohort in cohorts},
            "blocked": blocked,
            "projected_columns": "Partition membership, pre-treatment features and identity only; training labels only for provenance",
        },
    )
    frozen = json.loads(lock.read_text(encoding="utf-8"))
    results, support, paired = [], [], []
    write_json(
        target / "outcome_access.json",
        {
            "started_utc": now(),
            "split": designation,
            "action_freeze_sha256": checksum(lock),
            "cohorts": [item.name for item in cohorts],
        },
    )
    for cohort in cohorts:
        evaluate_locked(
            cohort,
            frozen["cohorts"][cohort.name],
            results,
            support,
            paired,
            designation,
        )
    comparisons = [] if fixture else validation_comparison(root, paired)
    for name, rows in (
        ("final_policy_evaluation", results),
        ("final_policy_support", support),
        ("final_paired_policy_comparisons", paired),
        ("validation_vs_test", comparisons),
    ):
        if rows:
            pd.DataFrame(rows).to_csv(target / f"{name}.csv", index=False)
    after = checksum(root / PROTOCOL)
    if after != digest:
        raise ValueError("Frozen protocol changed during evaluation")
    write_json(
        target / "frozen_protocol_verification.json",
        {"before": digest, "after": after, "unchanged": True},
    )
    summary = {
        "split": designation,
        "test_evaluated": not fixture and bool(cohorts),
        "completed_utc": now(),
        "protocol_sha256": digest,
        "seed": protocol["seed"],
        "cohorts_completed": [c.name for c in cohorts],
        "blocked": blocked,
        "runtime_seconds": perf_counter() - start,
        "action_freeze_sha256": checksum(lock),
        "uncertainty": INTERVAL_NOTE,
        "models": protocol["models"]
        if not fixture
        else {"scores": "deterministic fixture"},
    }
    write_json(
        target / "final_evaluation_manifest.json",
        summary
        | {"cohorts": frozen["cohorts"], "action_lock_utc": frozen["locked_utc"]},
    )
    write_json(target / "session_14_summary.json", summary)
    if comparisons:
        plots(target, pd.DataFrame(comparisons))
        primary = [
            row
            for row in paired
            if row["dataset"] in ("E", "Criteo 1M")
            and row["method"] == "dr"
            and row["fraction"] == 0.20
        ]
        write_json(
            target / "final_business_summary.json",
            {"primary": primary, "uncertainty": INTERVAL_NOTE},
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / "artifacts" / (
        "session_14_fixture" if args.fixture else "session_14"
    )
    if (
        args.fixture
        and destination.resolve() == (args.root / "artifacts/session_14").resolve()
    ):
        parser.error("Fixture output cannot use the real evaluation directory")
    summary = run(args.root, destination, args.fixture)
    print(json.dumps(summary, indent=2))
    if summary["blocked"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
