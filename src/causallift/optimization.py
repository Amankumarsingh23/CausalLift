"""Fixed-cost campaign economics and feasible binary allocation."""

from numbers import Integral
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from causallift.targeting import rank_scores


def _money(values):
    x = np.asarray(values, dtype=float)
    cents = np.rint(100 * x)
    if (
        not np.isfinite(x).all()
        or (x < 0).any()
        or (cents > 1e12).any()
        or not np.allclose(100 * x, cents, rtol=0, atol=1e-7)
    ):
        raise ValueError(
            "Costs and budget must be nonnegative finite money with at most two decimals"
        )
    return cents.astype(np.int64)


def economics(unit, effect, value, cost, eligible):
    frame = pd.DataFrame(
        {
            "unit_id": unit,
            "effect": effect,
            "value": value,
            "cost": cost,
            "eligible": eligible,
        }
    )
    if frame.unit_id.isna().any() or not frame.unit_id.is_unique:
        raise ValueError("Unique, nonmissing unit identifiers are required")
    if not pd.api.types.is_bool_dtype(frame.eligible) or frame.eligible.isna().any():
        raise ValueError("Eligibility must be boolean")
    if (
        not np.isfinite(frame[["effect", "value"]].to_numpy(dtype=float)).all()
        or (frame.value < 0).any()
    ):
        raise ValueError(
            "Finite effects and nonnegative contribution values are required"
        )
    _money(frame.cost)
    frame["incremental_value"] = frame.effect * frame.value
    frame["profit"] = frame.incremental_value - frame.cost
    frame["break_even"] = np.divide(
        frame.cost.to_numpy(dtype=float),
        frame.value.to_numpy(dtype=float),
        out=np.full(len(frame), np.nan),
        where=frame.value.to_numpy() > 0,
    )
    return frame


def discount_profit(mu0, mu1, value, discount, delivery):
    p0, p1, v, d, c = np.broadcast_arrays(
        *map(
            lambda a: np.asarray(a, dtype=float), (mu0, mu1, value, discount, delivery)
        )
    )
    if (
        not all(np.isfinite(a).all() for a in (p0, p1, v, d, c))
        or any(((p < 0) | (p > 1)).any() for p in (p0, p1))
        or any((x < 0).any() for x in (v, d, c))
    ):
        raise ValueError(
            "Valid probabilities and nonnegative economic inputs are required"
        )
    return (p1 - p0) * v - p1 * d - c


def allocate(
    profit,
    cost,
    eligible,
    budget,
    *,
    capacity=None,
    policy="milp",
    score=None,
    seed=42,
    groups=None,
    time_limit=1.0,
):
    """Return selected indices and diagnostics for an at-most-budget problem."""
    start = perf_counter()
    p = np.asarray(profit, dtype=float)
    c = _money(cost)
    allowed = np.asarray(eligible)
    if (
        p.ndim != 1
        or not np.isfinite(p).all()
        or c.shape != p.shape
        or allowed.shape != p.shape
        or allowed.dtype.kind != "b"
    ):
        raise ValueError(
            "Aligned finite profits, costs and boolean eligibility are required"
        )
    if np.ndim(budget) != 0:
        raise ValueError("Budget must be scalar")
    cash = int(_money(budget))
    if capacity is None:
        capacity = len(p)
    if (
        not isinstance(capacity, Integral)
        or isinstance(capacity, (bool, np.bool_))
        or capacity < 0
    ):
        raise ValueError("Capacity must be a nonnegative integer")
    if policy not in ("none", "ranking", "profit", "ratio", "uniform", "milp"):
        raise ValueError("Unknown allocation policy")
    if groups is not None:
        if policy != "milp":
            raise ValueError("Multi-treatment groups require MILP")
        g = np.asarray(groups)
        if g.shape != p.shape or pd.isna(g).any():
            raise ValueError("Action groups must align and be present")
    candidates = np.flatnonzero(
        allowed & (c <= cash) & ((p > 0) if policy not in ("ranking", "none") else True)
    )
    chosen = np.array([], dtype=int)
    status = "empty"
    optimal = policy in ("uniform", "milp")
    gap = bound = None
    if policy != "none" and capacity and len(candidates):
        if policy == "milp":
            cp, cc = p[candidates], c[candidates]
            rows = [np.zeros(len(cp), dtype=int), np.ones(len(cp), dtype=int)]
            counts = [cc, np.ones(len(cp))]
            upper = [cash, capacity]
            if groups is not None:
                codes, unique = pd.factorize(g[candidates], sort=False)
                rows.append(codes + 2)
                counts.append(np.ones(len(cp)))
                upper.extend([1] * len(unique))
            matrix = coo_matrix(
                (
                    np.concatenate(counts),
                    (np.concatenate(rows), np.tile(np.arange(len(cp)), len(rows))),
                ),
                shape=(len(upper), len(cp)),
            ).tocsc()
            res = milp(
                -cp,
                integrality=np.ones(len(cp)),
                bounds=Bounds(0, 1),
                constraints=LinearConstraint(matrix, 0, upper),
                options={"time_limit": time_limit, "mip_rel_gap": 0.0},
            )
            status = {
                0: "optimal",
                1: "limit",
                2: "infeasible",
                3: "unbounded",
                4: "solver_error",
            }.get(res.status, "unknown")
            optimal = res.status == 0
            if res.x is not None:
                rounded = np.rint(res.x)
                if np.max(np.abs(res.x - rounded)) <= 1e-5:
                    chosen = candidates[rounded.astype(bool)]
                else:
                    status += "_no_integer_solution"
                    optimal = False
            else:
                status += "_no_incumbent"
                optimal = False
            gap = getattr(res, "mip_gap", None)
            raw_bound = getattr(res, "mip_dual_bound", None)
            bound = (
                -raw_bound if raw_bound is not None and np.isfinite(raw_bound) else None
            )
            gap = float(gap) if gap is not None and np.isfinite(gap) else None
        else:
            if policy == "ranking":
                raw = np.asarray(score, dtype=float)
                if raw.shape != p.shape or not np.isfinite(raw).all():
                    raise ValueError("Ranking scores must be finite and aligned")
            elif policy == "ratio":
                raw = np.divide(p, np.maximum(c, 1), dtype=float)
                raw = np.where(c == 0, np.max(np.abs(raw)) + 1, raw)
            else:
                raw = p
            order = candidates[rank_scores(raw[candidates], seed=seed)]
            if policy == "uniform":
                if np.unique(c[allowed]).size > 1:
                    raise ValueError(
                        "Uniform optimization requires equal eligible costs"
                    )
                limit = (
                    capacity if c[order[0]] == 0 else min(capacity, cash // c[order[0]])
                )
                chosen = order[:limit]
            else:
                selection = []
                remaining = cash
                for i in order:
                    if c[i] <= remaining:
                        selection.append(i)
                        remaining -= int(c[i])
                        if len(selection) == capacity:
                            break
                chosen = np.asarray(selection, dtype=int)
            status = "exact_uniform" if policy == "uniform" else "heuristic"
    feasible = (
        len(chosen) <= capacity
        and int(c[chosen].sum()) <= cash
        and allowed[chosen].all()
        and (groups is None or len(np.unique(g[chosen])) == len(chosen))
    )
    if not feasible:
        raise RuntimeError(
            "Allocation failed exact cents, eligibility or capacity checks"
        )
    return chosen, {
        "status": status,
        "feasible": True,
        "proven_optimal": bool(optimal),
        "gap": gap,
        "bound": bound,
        "objective": float(p[chosen].sum()),
        "runtime_seconds": perf_counter() - start,
        "selected_count": len(chosen),
        "spend": int(c[chosen].sum()) / 100,
    }


def policy_totals(effect, value, cost, selected):
    a, v, c = map(np.asarray, (effect, value, cost))
    spend = float(c[selected].sum())
    incremental = float((a[selected] * v[selected]).sum())
    return {
        "conversions": float(a[selected].sum()),
        "value": incremental,
        "profit": incremental - spend,
        "roi": (incremental - spend) / spend if spend else None,
        "negative_share": float((a[selected] < 0).mean()) if len(selected) else None,
    }
