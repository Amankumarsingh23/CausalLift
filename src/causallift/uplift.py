"""Randomized evaluation metrics with distinct gain and Qini conventions."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
from scipy.stats import spearmanr

from causallift.experimentation import validated_rows
from causallift.targeting import rank_scores


def effect_accuracy(predicted, truth) -> dict:
    prediction, tau = np.asarray(predicted, dtype=float), np.asarray(truth, dtype=float)
    if (prediction.ndim != 1 or not prediction.size or tau.shape != prediction.shape
            or not np.isfinite(np.stack([prediction, tau])).all()):
        raise ValueError("Effects must be aligned nonempty finite vectors")
    error = prediction - tau
    constant = min(np.ptp(prediction), np.ptp(tau)) <= 1e-12
    signed = float(error.mean())
    return {"root_pehe": float(np.sqrt(np.mean(error**2))), "cate_mae": float(np.mean(abs(error))),
            "signed_ate_error": signed, "absolute_ate_error": abs(signed),
            "spearman": None if constant else float(spearmanr(prediction, tau).statistic)}


@dataclass(frozen=True)
class UpliftCurve:
    fraction: np.ndarray
    gain: np.ndarray
    random_gain: np.ndarray
    qini: np.ndarray
    random_qini: np.ndarray
    control_n: np.ndarray
    treated_n: np.ndarray
    control_events: np.ndarray
    treated_events: np.ndarray
    true_gain: np.ndarray | None

    def metrics(self) -> dict:
        valid = np.isfinite(self.qini)
        return {
            "auuc": float(np.trapezoid(self.gain, self.fraction)),
            "gain_area_above_random": float(np.trapezoid(self.gain - self.random_gain, self.fraction)),
            "qini_coefficient": float(np.trapezoid(
                (self.qini - self.random_qini)[valid], self.fraction[valid],
            )),
            "endpoint_gain": float(self.gain[-1]),
            "true_auuc": None if self.true_gain is None else float(np.trapezoid(self.true_gain, self.fraction)),
        }


class RandomizedEvaluation:
    def __init__(self, treatment, outcome, propensity):
        assigned, y = validated_rows(treatment, outcome, binary=True)
        if not assigned.any() or assigned.all():
            raise ValueError("Evaluation requires both assignment arms")
        e = np.broadcast_to(np.asarray(propensity, dtype=float), assigned.shape)
        if not np.all((0 < e) & (e < 1)):
            raise ValueError("Propensities must lie strictly between zero and one")
        self.assigned = assigned
        self.outcome = y
        self.pseudo_outcome = np.where(assigned, y / e, -y / (1 - e))
        if not np.isfinite(self.pseudo_outcome).all():
            raise ValueError("Propensities yield nonfinite pseudo-outcomes")
        self.n = len(y)
        self.endpoint = float(self.pseudo_outcome.sum() / self.n)

    def curve(self, scores, *, seed: int = 42, truth=None) -> UpliftCurve:
        return self.curve_from_order(rank_scores(scores, seed=seed), truth=truth)

    def curve_from_order(self, order, *, truth=None) -> UpliftCurve:
        order = np.asarray(order)
        if (order.shape != (self.n,) or order.dtype.kind not in "iu" or (order < 0).any()
                or (order >= self.n).any()
                or not np.all(np.bincount(order.astype(np.intp, copy=False), minlength=self.n) == 1)):
            raise ValueError("Ranking must include each evaluation row exactly once")
        assigned, y = self.assigned[order], self.outcome[order]
        k = np.arange(self.n + 1)
        fraction = k / self.n
        nt = np.r_[0, np.cumsum(assigned)]
        nc = k - nt
        yt = np.r_[0, np.cumsum(assigned * y)]
        yc = np.r_[0, np.cumsum(~assigned * y)]
        gain = np.r_[0., np.cumsum(self.pseudo_outcome[order])] / self.n
        gain[-1] = self.endpoint
        qini = np.full(self.n + 1, np.nan)
        eligible = (nt > 0) & (nc > 0)
        qini[eligible] = (yt[eligible] - yc[eligible] * nt[eligible] / nc[eligible]) / self.n
        qini[0] = 0
        true_gain = None
        if truth is not None:
            tau = np.asarray(truth, dtype=float)
            if tau.shape != (self.n,) or not np.all(np.abs(tau) <= 1):
                raise ValueError("Truth must contain one finite effect per evaluation row")
            true_gain = np.r_[0., np.cumsum(tau[order])] / self.n
            true_gain[-1] = tau.mean()
        return UpliftCurve(fraction, gain, fraction * self.endpoint, qini, fraction * qini[-1],
                           nc, nt, yc, yt, true_gain)

    def summarize_selection(self, selected, *, minimum: int = 1) -> dict:
        t, y, z = self.assigned[selected], self.outcome[selected], self.pseudo_outcome[selected]
        nt, nc = int(t.sum()), int((~t).sum())
        yt, yc = int(y[t].sum()), int(y[~t].sum())
        warning = ""
        if min(nt, nc) < minimum:
            warning = "Insufficient treatment/control support"
        elif minimum > 1 and yt + yc < minimum:
            warning = "Too few outcome events"
        estimate = None if warning else float(z.mean())
        se = float(z.std(ddof=1) / sqrt(len(z))) if estimate is not None and yt + yc else None
        if not warning and yt + yc == 0:
            warning = "Zero observed events; no normal interval"
        elif not warning and min(yt, yc) < 5:
            warning = "Sparse arm event counts; normal interval can be unreliable"
        return {
            "n": len(z), "treated_n": nt, "control_n": nc, "treated_events": yt, "control_events": yc,
            "uplift": estimate, "gain": float(z.sum() / self.n),
            "effect_se": se, "ci_low": estimate - 1.96 * se if se is not None else None,
            "ci_high": estimate + 1.96 * se if se is not None else None, "warning": warning,
        }

    def calibration(self, scores, *, bins: int = 10, truth=None) -> list[dict]:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (self.n,) or not np.isfinite(scores).all() or bins < 1:
            raise ValueError("Calibration requires finite aligned scores and at least one bin")
        if truth is not None:
            truth = np.asarray(truth, dtype=float)
            if truth.shape != scores.shape or not np.all(np.abs(truth) <= 1):
                raise ValueError("Calibration truth must align with scores")
        cuts = np.unique(np.quantile(scores, np.linspace(0, 1, bins + 1)[1:-1]))
        labels = np.searchsorted(cuts, scores, side="left")
        rows = []
        for i, label in enumerate(np.unique(labels), start=1):
            mask = labels == label
            row = self.summarize_selection(mask, minimum=5)
            rows.append(row | {"bin": i, "mean_predicted_cate": float(scores[mask].mean()),
                               "mean_true_cate": None if truth is None else float(truth[mask].mean())})
        return rows
