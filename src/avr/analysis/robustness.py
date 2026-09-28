"""Load robustness-suite results and turn them into tables and figures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from avr.eval.stats import holm, mcnemar, wilson_ci
from avr.perturb.suite import FACTORS

# Categorical order from the validated reference palette (light surface);
# one slot per policy, assigned in fixed order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

FACTOR_ORDER = [
    "light_intensity", "light_color", "light_direction", "table_color", "background",
    "distractors", "camera_pose", "noise", "blur", "jpeg", "cube_color",
]  # fmt: skip


def load_cells(root: str | Path) -> dict[tuple[str, int], dict]:
    """{(factor, level): cell} for every finished cell under `root`."""
    cells = {}
    for path in Path(root).glob("*/level*.json"):
        cell = json.loads(path.read_text())
        cells[(cell["factor"] or "clean", cell["level"])] = cell
    return cells


def summary_rows(cells: dict, clean: dict | None = None) -> list[dict]:
    """One row per cell: success rate, 95% CI and where failures stop.

    With `clean` (the unperturbed cell of the same policy, covering the same
    seeds), each row also gets a paired comparison on exactly those episodes:
    the clean success rate on the same seeds, the difference, and an exact
    McNemar p-value, Holm-corrected over all rows.
    """
    rows = []
    items = sorted(cells.items(), key=lambda kv: (_factor_rank(kv[0][0]), kv[0][1]))
    clean_by_seed = {e["seed"]: e["success"] for e in clean["episodes"]} if clean else {}
    for (factor, level), cell in items:
        eps = cell["episodes"]
        n, k = len(eps), sum(e["success"] for e in eps)
        lo, hi = wilson_ci(k, n)
        stages = Counter(int(round(e["max_reward"])) for e in eps if not e["success"])
        row = {
            "factor": factor, "level": level,
            "category": FACTORS[factor].category if factor in FACTORS else "clean",
            "n": n, "success": k / n, "ci_lo": lo, "ci_hi": hi,
            "fail_no_touch": stages.get(0, 0) / n,
            "fail_grasp": stages.get(1, 0) / n,
            "fail_transport": (stages.get(2, 0) + stages.get(3, 0)) / n,
        }  # fmt: skip
        if clean_by_seed and factor != "clean":
            paired = [(clean_by_seed[e["seed"]], e["success"]) for e in eps if e["seed"] in clean_by_seed]
            if len(paired) != n:
                raise ValueError(f"clean run misses seeds of {factor} level {level}")
            test = mcnemar([a for a, _ in paired], [b for _, b in paired])
            clean_rate = sum(a for a, _ in paired) / n
            row.update(clean_same_seeds=clean_rate, delta=k / n - clean_rate,
                       retained=(k / n) / clean_rate if clean_rate else float("nan"),
                       only_clean=test["only_a"], only_perturbed=test["only_b"], p=test["p_value"])
        rows.append(row)
    tested = [r for r in rows if "p" in r]
    for r, p_adj in zip(tested, holm([r["p"] for r in tested])):
        r["p_holm"] = p_adj
    return rows


def _factor_rank(name: str) -> int:
    if name == "clean":
        return -1
    return FACTOR_ORDER.index(name) if name in FACTOR_ORDER else len(FACTOR_ORDER)


def _clean_rate_on(cell: dict, clean_cell: dict) -> float:
    by_seed = {e["seed"]: e["success"] for e in clean_cell["episodes"]}
    return sum(by_seed[e["seed"]] for e in cell["episodes"]) / len(cell["episodes"])


def plot_robustness(results: dict[str, dict], clean_rates: dict[str, float] | None = None, path=None,
                    relative_to: dict[str, dict] | None = None):
    """Small multiples, one panel per factor: success rate vs. level.

    `results` maps a policy label to its `load_cells()` dict; level 0 of each
    curve is that policy's clean success rate (`clean_rates`), drawn when given.

    With `relative_to` ({label: that policy's clean cell}) every point is divided
    by the policy's clean success rate *on the same seeds*: 100 % = no loss. This
    compares robustness of policies with different clean performance. The band is
    the Wilson CI scaled by the same factor (ignores the clean run's own noise).
    """
    import matplotlib.pyplot as plt

    factors = [f for f in FACTOR_ORDER if any((f, lv) in c for c in results.values() for lv in range(1, 5))]
    ncols = 4
    nrows = -(-len(factors) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.5 * nrows), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel()

    for ax, factor in zip(axes, factors):
        ax.set_facecolor(SURFACE)
        for i, (label, cells) in enumerate(results.items()):
            color = SERIES_COLORS[i]
            pts = []
            if relative_to and label in relative_to:
                pts.append((0, 1.0, None, None))
            elif clean_rates and label in clean_rates:
                pts.append((0, clean_rates[label], None, None))
            for level in range(1, 5):
                if (factor, level) in cells:
                    eps = cells[(factor, level)]["episodes"]
                    k, n = sum(e["success"] for e in eps), len(eps)
                    lo, hi = wilson_ci(k, n)
                    scale = 1.0
                    if relative_to and label in relative_to:
                        scale = 1 / max(_clean_rate_on(cells[(factor, level)], relative_to[label]), 1e-9)
                    pts.append((level, k / n * scale, lo * scale, hi * scale))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ci = [p for p in pts if p[2] is not None]
            ax.fill_between([p[0] for p in ci], [p[2] for p in ci], [p[3] for p in ci],
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        cat = FACTORS[factor].category
        ax.set_title(factor.replace("_", " ") + (" (task-relevant)" if cat == "task" else ""),
                     fontsize=9.5, color=TEXT_PRIMARY, loc="left")
        if relative_to:
            ax.axhline(1.0, color=TEXT_SECONDARY, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.set_ylim(0, 2.0)
        else:
            ax.set_ylim(0, 1.02)
        ax.set_xticks(range(5))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    for ax in axes[len(factors):]:
        ax.set_visible(False)
    for ax in axes[::ncols]:
        ax.set_ylabel("success / clean (same seeds)" if relative_to else "success rate", fontsize=8.5,
                      color=TEXT_SECONDARY)
    for ax in axes[max(0, len(factors) - ncols):len(factors)]:  # lowest panel of each column
        ax.set_xlabel("severity level", fontsize=8.5, color=TEXT_SECONDARY)
        ax.xaxis.set_tick_params(labelbottom=True)
    if len(results) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncols=len(results))
    title = ("Success retained under visual perturbations, relative to each policy's clean rate on the same seeds"
             if relative_to else "Success rate under visual perturbations (shaded: 95% CI)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig
