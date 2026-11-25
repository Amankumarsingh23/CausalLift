"""Render final figures from the locked aggregate comparisons."""

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plots(target, frame):
    primary = frame.loc[frame.dataset.isin(("E", "Criteo 1M"))].copy()
    if primary.empty:
        return
    primary["label"] = primary.dataset + " vs. " + primary.reference
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), layout="constrained")
    for ax, (name, group) in zip(
        axes, primary.groupby("dataset", sort=False), strict=False
    ):
        point = group.test_incremental_profit
        ax.errorbar(
            point,
            group.reference,
            xerr=[point - group.test_ci_profit_low, group.test_ci_profit_high - point],
            fmt="o",
            color="#087e76",
            capsize=4,
        )
        ax.axvline(0, color="#76828c", linewidth=1)
        ax.set(
            title=f"{name} · reserved test",
            xlabel="Incremental profit (simulated or hypothetical units)",
        )
    fig.savefig(target / "final_policy_comparison.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.35), layout="constrained")
    for ax, (name, group) in zip(
        axes, primary.groupby("dataset", sort=False), strict=False
    ):
        for row in group.itertuples():
            ax.plot(
                ["Validation", "Reserved test"],
                [row.validation_profit_per_unit, row.test_profit_per_unit],
                marker="o",
                label=f"vs. {row.reference}",
            )
        ax.axhline(0, color="#76828c", linewidth=1)
        ax.legend()
        ax.set(
            title=name, ylabel="Estimated profit per eligible or ineligible cohort unit"
        )
    fig.savefig(target / "validation_vs_test.png", dpi=155)
    plt.close(fig)
