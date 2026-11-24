import json

import numpy as np
import pandas as pd

from causallift.observational import match
from causallift.session10 import (
    add_noise,
    continuous_design,
    estimate,
    position_digest,
    reliability_status,
    run,
    shuffle_assignments,
    subset_positions,
)


def test_balanced_nearest_ties_and_caliper():
    t = np.r_[np.zeros(4), np.ones(9)]
    a = match(t, np.full(len(t), 0.5))
    np.testing.assert_array_equal(np.bincount(a[1]), [3, 2, 2, 2])
    np.testing.assert_array_equal(a[1], match(t, np.full(len(t), 0.5))[1])
    p = np.r_[[0.25, 0.25, 0.75, 0.75], np.full(8, 0.5), 0.99]
    chosen = match(t, p, caliper_sd=1.05)
    assert len(chosen[0]) == 8
    assert np.bincount(chosen[1]).tolist() == [2, 2, 2, 2]
    legacy = match(t, p, caliper_sd=1.05, legacy_ties=True)
    np.testing.assert_array_equal(chosen[0], legacy[0])
    np.testing.assert_array_equal(chosen[2], legacy[2])


def test_placebo_refits_and_preserves_counts(monkeypatch):
    from causallift import session10 as mod

    x = pd.DataFrame({"x": np.arange(40)})
    t = np.tile([0, 1], 20)
    y = np.tile([0, 0, 1, 1], 10)
    perm = shuffle_assignments(t, np.random.default_rng(11))
    assert perm.sum() == t.sum() and not np.array_equal(t, perm)
    calls = []
    original = mod.MetaLearner.fit
    propensity = mod.fit_binary_outcome

    def spy(self, X, T, Y):
        calls.append(np.asarray(T).copy())
        return original(self, X, T, Y)

    def spy_propensity(X, T, *args):
        calls.append(np.asarray(T).copy())
        return propensity(X, T, *args)

    monkeypatch.setattr(mod.MetaLearner, "fit", spy)
    monkeypatch.setattr(mod, "fit_binary_outcome", spy_propensity)
    estimate(x, perm, y, x, t, y, seed=11)
    assert len(calls) == 2
    for seen in calls:
        np.testing.assert_array_equal(seen, perm)


def test_noise_subsets_and_simulation_truth():
    t = np.tile([0, 1], 20)
    x = pd.DataFrame({"x": np.arange(40)})
    a, b = add_noise(x, x, np.random.default_rng(11))
    pd.testing.assert_frame_equal(a, add_noise(x, x, np.random.default_rng(11))[0])
    assert not a["__independent_noise"].equals(b["__independent_noise"])
    keep = subset_positions(t, np.random.default_rng(3))
    assert len(keep) == 32 and t[keep].sum() == 16 and len(np.unique(keep)) == 32
    assert position_digest(keep) == position_digest(
        subset_positions(t, np.random.default_rng(3))
    )
    frame, treat, y, nc, p = continuous_design(np.random.default_rng(4), 100, 1.6)
    z, u, err, ncerr = np.random.default_rng(4).normal(size=(4, 100))
    np.testing.assert_allclose(y - (0.9 * z + 1.6 * u + err), treat, atol=1e-14)
    np.testing.assert_allclose(nc, 0.9 * z + 1.6 * u + ncerr)
    np.testing.assert_allclose(frame.u, u)
    assert np.all((0 < p) & (p < 1))


def test_fixture_estimands_and_reliability(tmp_path, monkeypatch):
    from causallift import session10 as mod

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixture does not load persisted partitions")

    monkeypatch.setattr(mod, "load_dataset", forbidden)
    run(tmp_path, fixture=True)
    out = tmp_path / "artifacts/session_10_fixture"
    report = json.loads((out / "causal_reliability_report.json").read_text())[
        "datasets"
    ]
    assert report["F"]["status"] == "not_identified"
    assert reliability_status(False, None) == "not_identified"
    assert reliability_status(None, False) == "not_assessed"
    assert report["F"]["placebo"] is None
    assert not report["A"]["extreme_weight_warning"]
    subsets = pd.read_csv(out / "subset_stability.csv")
    assert subsets.target.str.contains("subset ATE").all()
    assert (subsets.n < subsets.train_n).all()
    assert subsets.validation_units_sha256.str.len().eq(64).all()
    profiles = pd.read_csv(out / "overlap_sensitivity.csv")
    assert profiles.ess_treated_fraction.between(0, 1 + 1e-9).all()
    assert (out / "sensitivity_analysis.png").is_file()
