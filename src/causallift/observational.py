"""Estimators for binary assignment and finite, real-valued outcomes."""

from math import sqrt

import numpy as np
from scipy.special import logit

from causallift.experimentation import validated_rows


def design(treatment, probability):
    t, _ = validated_rows(treatment, treatment, binary=True)
    e = np.broadcast_to(np.asarray(probability, dtype=float), t.shape)
    if not t.any() or t.all() or not np.all((0 < e) & (e < 1)):
        raise ValueError("Both arms and strictly interior propensities are required")
    return t, e


def standardize(mu0, mu1, mask=None):
    a, b = np.asarray(mu0), np.asarray(mu1)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite([a, b]).all():
        raise ValueError("Outcome predictions must be aligned and finite")
    delta = b - a
    selected = delta if mask is None else delta[mask]
    if not len(selected):
        raise ValueError("Empty standardization target")
    return float(selected.mean())


def weights(treatment, probability, kind="ate"):
    t, e = design(treatment, probability)
    if kind == "ate":
        w = np.where(t, 1 / e, 1 / (1 - e))
    elif kind == "stabilized":
        share = t.mean()
        w = np.where(t, share / e, (1 - share) / (1 - e))
    elif kind == "att":
        w = np.where(t, 1, e / (1 - e))
    else:
        raise ValueError("Unknown weight type")
    if not np.isfinite(w).all():
        raise ValueError("Propensities imply nonfinite weights")
    return w


def weighted_contrast(t, y, w):
    t, y = validated_rows(t, y, binary=False)
    w = np.asarray(w, dtype=float)
    if (
        w.shape != y.shape
        or not np.isfinite(w).all()
        or (w < 0).any()
        or min(w[t].sum(), w[~t].sum()) <= 0
    ):
        raise ValueError("Both weighted arms must have positive finite mass")
    return float(np.average(y[t], weights=w[t]) - np.average(y[~t], weights=w[~t]))


def ht(treatment, outcome, probability):
    t, y = validated_rows(treatment, outcome, binary=False)
    return float(np.mean(np.where(t, y, -y) * weights(t, probability)))


def effective_n(weights):
    weights = np.asarray(weights, dtype=float)
    if (
        not weights.size
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.max() == 0
    ):
        raise ValueError("ESS requires nonnegative finite weights with positive mass")
    w = weights / weights.max()
    return float(w.sum() ** 2 / np.sum(w**2))


def aipw(treatment, outcome, probability, mu0, mu1, *, att=False):
    t, y = validated_rows(treatment, outcome, binary=False)
    _, e = design(t, probability)
    a, b = np.asarray(mu0, dtype=float), np.asarray(mu1, dtype=float)
    if a.shape != y.shape or b.shape != y.shape or not np.isfinite([a, b]).all():
        raise ValueError("Outcome nuisances must be finite and aligned")
    if att:
        numerator = np.where(t, y - a, -e / (1 - e) * (y - a))
        point = numerator.mean() / t.mean()
        influence = (numerator - point * t) / t.mean()
    else:
        score = b - a + np.where(t, (y - b) / e, -(y - a) / (1 - e))
        point = score.mean()
        influence = score - point
    if not np.isfinite(influence).all() or not np.isfinite(point):
        raise ValueError("AIPW produced nonfinite scores")
    variance = float(influence.var(ddof=1))
    se = sqrt(variance / len(y))
    return {
        "estimate": float(point),
        "score_variance": variance,
        "se": se,
        "ci_low": float(point - 1.96 * se),
        "ci_high": float(point + 1.96 * se),
    }


def match(treatment, probability, *, caliper_sd=0.2, legacy_ties=False):
    """Match with replacement, balancing reuse among tied nearest controls."""
    t, e = design(treatment, probability)
    if not np.isfinite(caliper_sd) or caliper_sd < 0:
        raise ValueError("Caliper must be finite and nonnegative")
    scores = logit(e)
    radius = float(caliper_sd * scores.std(ddof=1))
    treated, controls = np.flatnonzero(t), np.flatnonzero(~t)
    sorted_controls = controls[np.argsort(scores[controls], kind="stable")]
    values, first, sizes = np.unique(
        scores[sorted_controls], return_index=True, return_counts=True
    )
    split = np.searchsorted(values, scores[treated])
    left = np.maximum(split - 1, 0)
    right = np.minimum(split, len(values) - 1)
    dl, dr = abs(scores[treated] - values[left]), abs(scores[treated] - values[right])
    distance = np.minimum(dl, dr)
    keep = distance <= radius
    if legacy_ties:
        nearest = np.where(dl <= dr, left, right)
        return (
            treated[keep],
            sorted_controls[first[nearest[keep]]],
            distance[keep],
            radius,
        )
    reuse = np.zeros(len(values), dtype=int)
    chosen = []
    for a, b, da, db in zip(left[keep], right[keep], dl[keep], dr[keep], strict=True):
        if da == db:
            group = a if reuse[a] // sizes[a] <= reuse[b] // sizes[b] else b
        else:
            group = a if da < db else b
        chosen.append(sorted_controls[first[group] + reuse[group] % sizes[group]])
        reuse[group] += 1
    return treated[keep], np.asarray(chosen, dtype=int), distance[keep], radius


def overlap_profiles(probability):
    p = np.asarray(probability, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("Expected aligned finite probabilities in [0, 1]")
    return {
        "unmodified": (np.ones(len(p), dtype=bool), p),
        "clip_05_95": (np.ones(len(p), dtype=bool), np.clip(p, 0.05, 0.95)),
        "restrict_05_95": ((p >= 0.05) & (p <= 0.95), p),
    }


def weighted_smd(t, x, w, *, denominator):
    t, x = validated_rows(t, x, binary=False)
    if not np.isfinite(denominator) or denominator < 0:
        raise ValueError("Invalid standardization denominator")
    if denominator == 0 or min(np.sum(w[t]), np.sum(w[~t])) == 0:
        return None
    return weighted_contrast(t, x, w) / denominator
