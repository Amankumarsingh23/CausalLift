"""Assignment checks, pre-treatment balance, and multiplicity adjustments."""

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from causallift.data.contract import DatasetSpec
from causallift.experimentation import validated_rows
from causallift.planning import validate_alpha, validate_counts


@dataclass(frozen=True)
class SRMResult:
    control_n: int
    treated_n: int
    expected_treatment_share: float | None
    observed_treatment_share: float
    observed_control_share: float
    expected_control_n: float | None
    expected_treated_n: float | None
    chi_square: float | None
    degrees_of_freedom: int | None
    p_value: float | None
    alpha: float
    alert: bool | None
    status: str
    method: str | None
    warnings: tuple[str, ...] = ()


def sample_ratio_mismatch(
    control_n: int, treated_n: int, expected_treatment_share: float | None = None, *, alpha: float = 0.01,
) -> SRMResult:
    validate_counts(control_n, treated_n, minimum=0)
    validate_alpha(alpha)
    total = int(control_n) + int(treated_n)
    if not total:
        raise ValueError("SRM requires at least one observed assignment")
    common = dict(control_n=int(control_n), treated_n=int(treated_n),
                  expected_treatment_share=expected_treatment_share,
                  observed_treatment_share=treated_n / total, observed_control_share=control_n / total, alpha=alpha)
    if expected_treatment_share is None:
        return SRMResult(**common, expected_control_n=None, expected_treated_n=None, chi_square=None,
                         degrees_of_freedom=None, p_value=None, alert=None, status="not_assessable", method=None,
                         warnings=("Intended assignment probabilities are unavailable.",))
    if not 0 < expected_treatment_share < 1:
        raise ValueError("Expected treatment share must be strictly between zero and one")
    ec, et = total * (1 - expected_treatment_share), total * expected_treatment_share
    stat = (control_n - ec) ** 2 / ec + (treated_n - et) ** 2 / et
    if min(ec, et) < 5:
        method, warnings = "exact_binomial", ("An expected assignment count is below five.",)
        p = binomtest(int(treated_n), total, expected_treatment_share).pvalue
    else:
        method, warnings = "pearson_chi_square", ()
        p = float(chi2.sf(stat, 1))
    return SRMResult(**common, expected_control_n=ec, expected_treated_n=et,
                     chi_square=stat, degrees_of_freedom=1, p_value=p, alert=p < alpha,
                     status="mismatch" if p < alpha else "no_mismatch_detected", method=method, warnings=warnings)


def covariate_balance(
    frame: pd.DataFrame, spec: DatasetSpec, features: tuple[str, ...], *, threshold: float = 0.1,
) -> list[dict]:
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The absolute SMD threshold must be finite and positive")
    if not features or len(set(features)) != len(features) or not set(features).issubset(spec.feature_columns):
        raise ValueError("Choose unique allowlisted pre-treatment features")
    rows = []
    for feature in features:
        assigned, values = validated_rows(frame[spec.treatment_column], frame[feature], binary=False)
        values = values.astype(float, copy=False)
        control, treated = values[~assigned], values[assigned]
        validate_counts(len(control), len(treated), minimum=2)
        mc, mt = float(control.mean()), float(treated.mean())
        vc, vt = float(control.var(ddof=1)), float(treated.var(ddof=1))
        if not np.isfinite([mc, mt, vc, vt, mt - mc]).all():
            raise ValueError(f"Balance summaries exceed floating-point range for {feature}")
        denominator = sqrt(vc / 2 + vt / 2)
        smd = (mt - mc) / denominator if denominator else None
        warnings = "Both arms have zero variance. SMD is undefined." if denominator == 0 else (
            "One arm has zero variance." if min(vc, vt) == 0 else ""
        )
        rows.append({"dataset": spec.dataset_id, "feature": feature, "control_n": len(control),
                     "treated_n": len(treated), "control_mean": mc, "treated_mean": mt, "smd": smd,
                     "absolute_smd": abs(smd) if smd is not None else None, "threshold": threshold,
                     "flag": abs(smd) >= threshold if smd is not None else (True if mt != mc else None),
                     "warnings": warnings})
    return rows


@dataclass(frozen=True)
class MultiplicityResult:
    raw_p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    reject: tuple[bool, ...]
    method: str
    alpha: float
    family_size: int
    error_rate: str


def adjust_p_values(p_values, method: str, *, alpha: float = 0.05) -> MultiplicityResult:
    validate_alpha(alpha)
    if method not in {"bonferroni", "holm", "benjamini_hochberg"}:
        raise ValueError("Use bonferroni, holm, or benjamini_hochberg")
    p = np.asarray(p_values)
    if p.ndim != 1 or not p.size or p.dtype.kind not in "iuf" or not np.all((0 <= p) & (p <= 1)):
        raise ValueError("The family must be a nonempty vector of finite p-values in [0, 1]")
    size = len(p)
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    if method == "bonferroni":
        corrected = ranked * size
        reject_ranked = ranked <= alpha / size
    elif method == "holm":
        remaining = np.arange(size, 0, -1)
        corrected = np.maximum.accumulate(ranked * remaining)
        reject_ranked = np.logical_and.accumulate(ranked <= alpha / remaining)
    else:
        fractions = np.arange(1, size + 1) / size
        scaled = ranked / fractions
        corrected = np.minimum.accumulate(scaled[::-1])[::-1]
        reject_ranked = np.logical_or.accumulate((ranked <= alpha * fractions)[::-1])[::-1]
    inverse_order = np.argsort(order)
    adjusted = np.minimum(1, corrected)[inverse_order]
    return MultiplicityResult(tuple(map(float, p)), tuple(map(float, adjusted)),
                              tuple(map(bool, reject_ranked[inverse_order])), method, alpha, size,
                              "FDR" if method == "benjamini_hochberg" else "FWER")
