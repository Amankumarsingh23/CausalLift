"""Offline value scores for deterministic binary policies on a common cohort."""

import numpy as np
import pandas as pd

from causallift.experimentation import validated_rows
from causallift.observational import effective_n

METHODS = ("dm", "ips", "snips", "dr")
INTERVAL_NOTE = "Fixed-action conditional Wald approximation; excludes reoptimization, nuisance-fit and trial clustering uncertainty."


class PolicyEvaluation:
    def __init__(
        self, units, treatment, outcome, propensity, mu0, mu1, *, contribution=1, cost=0
    ):
        self.t, self.y = validated_rows(treatment, outcome, binary=True)
        self.n = len(self.t)
        identities = pd.Index(units)
        if len(identities) != self.n or not identities.is_unique or identities.hasnans:
            raise ValueError(
                "Unique, nonmissing unit identifiers must align with outcomes"
            )
        self.units = identities.to_numpy()
        self.e = self._numeric(propensity, "propensity")
        self.mu0, self.mu1 = self._numeric(mu0, "mu0"), self._numeric(mu1, "mu1")
        if not np.all((0 < self.e) & (self.e < 1)):
            raise ValueError("Logging propensities must be strictly interior")
        if any(not np.all((0 <= p) & (p <= 1)) for p in (self.mu0, self.mu1)):
            raise ValueError("Outcome nuisances must be probabilities in [0, 1]")
        self.value, self.cost = (
            self._numeric(contribution, "contribution"),
            self._numeric(cost, "cost"),
        )
        if (self.value < 0).any() or (self.cost < 0).any():
            raise ValueError("Economics must be nonnegative")
        self.logged_probability = np.where(self.t, self.e, 1 - self.e)
        self.logged_mean = np.where(self.t, self.mu1, self.mu0)

    def _numeric(self, x, name):
        p = np.broadcast_to(np.asarray(x, dtype=float), (self.n,))
        if not np.isfinite(p).all():
            raise ValueError(f"{name} must be finite")
        return p

    def action(self, action):
        a = np.asarray(action)
        if (
            a.shape != (self.n,)
            or a.dtype.kind not in "bifu"
            or not np.isin(a, [0, 1]).all()
        ):
            raise ValueError("Policy action must be an aligned binary vector")
        return a.astype(bool)

    def scores(self, action):
        a = self.action(action)
        w = (self.t == a) / self.logged_probability
        dm = np.where(a, self.mu1, self.mu0)
        ips = w * self.y
        dr = dm + w * (self.y - self.logged_mean)
        if not np.isfinite([w, dr]).all():
            raise ValueError("Nonfinite evaluation weights or scores")
        return {"dm": dm, "ips": ips, "dr": dr}, w

    def support(self, a, w):
        matches = int(np.count_nonzero(w))
        positive = w[w > 0]
        ess = effective_n(positive) if matches else 0.0
        missing = any(
            np.any(a == i) and not np.any((a == i) & (self.t == i))
            for i in (False, True)
        )
        warnings = []
        if missing:
            warnings.append("A proposed action has no matching logged rows.")
        if ess < 0.2 * self.n or w.max() > 20:
            warnings.append("Concentrated weights or low ESS.")
        return {
            "matched_n": matches,
            "match_fraction": matches / self.n,
            "ess": ess,
            "max_weight": float(w.max()),
            "mean_weight": float(w.mean()),
            "positive_weight_p95": float(np.quantile(positive, 0.95))
            if matches
            else None,
            "matched_treated": int((a & self.t).sum()),
            "matched_control": int((~a & ~self.t).sum()),
            "missing_action_support": missing,
            "support_warning": " ".join(warnings),
        }

    def compare(self, action, reference=None, *, intervals=True):
        a = self.action(action)
        b = (
            np.zeros(self.n, dtype=bool)
            if reference is None
            else self.action(reference)
        )
        score, w = self.scores(a)
        baseline, u = self.scores(b)
        support = self.support(a, w)
        baseline_support = self.support(b, u)
        spend = float(self.cost[a].sum())
        ref_spend = float(self.cost[b].sum())
        incremental_cost = self.cost * (a.astype(int) - b.astype(int))
        rows = []
        identical = np.array_equal(a, b)
        for method in METHODS:
            if method == "snips":
                va = float(score["ips"].sum() / w.sum()) if w.sum() > 0 else None
                vb = float(baseline["ips"].sum() / u.sum()) if u.sum() > 0 else None
                delta = None if va is None or vb is None else va - vb
                total = None
                uncertainty = {}
            else:
                va = float(score[method].mean())
                vb = float(baseline[method].mean())
                diff = score[method] - baseline[method]
                supported = method == "dm" or not (
                    support["missing_action_support"]
                    or baseline_support["missing_action_support"]
                )
                if method == "ips":
                    if not w.any():
                        va = None
                    if not u.any():
                        vb = None
                delta = (
                    float(diff.mean()) if va is not None and vb is not None else None
                )
                profit = self.value * diff - incremental_cost
                total = float(profit.sum()) if delta is not None else None
                uncertainty = {}
                if intervals and supported and self.n > 1 and delta is not None:
                    se = float(diff.std(ddof=1) / np.sqrt(self.n))
                    pse = float(profit.std(ddof=1) * np.sqrt(self.n))
                    uncertainty = {
                        "se_rate": se,
                        "ci_rate_low": delta - 1.96 * se,
                        "ci_rate_high": delta + 1.96 * se,
                        "se_profit_total": pse,
                        "ci_profit_low": total - 1.96 * pse,
                        "ci_profit_high": total + 1.96 * pse,
                    }
            rows.append(
                {
                    "method": method,
                    "n": self.n,
                    "selected": int(a.sum()),
                    "spend": spend,
                    "reference_spend": ref_spend,
                    "value_rate": va,
                    "reference_rate": vb,
                    "incremental_rate": 0.0 if identical else delta,
                    "incremental_conversions": 0.0
                    if identical
                    else None
                    if delta is None
                    else delta * self.n,
                    "incremental_contribution": None
                    if total is None
                    else total + spend - ref_spend,
                    "incremental_profit": 0.0
                    if identical and method != "snips"
                    else total,
                    "roi": total / spend
                    if total is not None and spend and not b.any()
                    else None,
                    "reference_missing_support": baseline_support[
                        "missing_action_support"
                    ],
                    "interval_scope": INTERVAL_NOTE if uncertainty else "unavailable",
                    **support,
                    **uncertainty,
                }
            )
        return rows
