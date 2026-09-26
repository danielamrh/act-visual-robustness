"""Load robustness-suite results and turn them into tables and figures."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from avr.eval.stats import wilson_ci
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
    """One row per cell: success rate, 95% CI and where failures stop."""
    rows = []
    items = sorted(cells.items(), key=lambda kv: (_factor_rank(kv[0][0]), kv[0][1]))
    for (factor, level), cell in items:
        eps = cell["episodes"]
        n, k = len(eps), sum(e["success"] for e in eps)
        lo, hi = wilson_ci(k, n)
        stages = Counter(int(round(e["max_reward"])) for e in eps if not e["success"])
        rows.append({
            "factor": factor, "level": level,
            "category": FACTORS[factor].category if factor in FACTORS else "clean",
            "n": n, "success": k / n, "ci_lo": lo, "ci_hi": hi,
            "fail_no_touch": stages.get(0, 0) / n,
            "fail_grasp": stages.get(1, 0) / n,
            "fail_transport": (stages.get(2, 0) + stages.get(3, 0)) / n,
        })  # fmt: skip
    return rows


def _factor_rank(name: str) -> int:
    return FACTOR_ORDER.index(name) if name in FACTOR_ORDER else -1


def plot_robustness(results: dict[str, dict], clean_rates: dict[str, float] | None = None, path=None):
    """Small multiples, one panel per factor: success rate vs. level.

    `results` maps a policy label to its `load_cells()` dict; level 0 of each
    curve is that policy's clean success rate (`clean_rates`), drawn when given.
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
            if clean_rates and label in clean_rates:
                pts.append((0, clean_rates[label], None, None))
            for level in range(1, 5):
                if (factor, level) in cells:
                    eps = cells[(factor, level)]["episodes"]
                    k, n = sum(e["success"] for e in eps), len(eps)
                    lo, hi = wilson_ci(k, n)
                    pts.append((level, k / n, lo, hi))
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            ci = [p for p in pts if p[2] is not None]
            ax.fill_between([p[0] for p in ci], [p[2] for p in ci], [p[3] for p in ci],
                            color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        cat = FACTORS[factor].category
        ax.set_title(factor.replace("_", " ") + (" (task-relevant)" if cat == "task" else ""),
                     fontsize=9.5, color=TEXT_PRIMARY, loc="left")
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
        ax.set_ylabel("success rate", fontsize=8.5, color=TEXT_SECONDARY)
    for ax in axes[max(0, len(factors) - ncols):len(factors)]:  # lowest panel of each column
        ax.set_xlabel("severity level", fontsize=8.5, color=TEXT_SECONDARY)
        ax.xaxis.set_tick_params(labelbottom=True)
    if len(results) > 1:
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9, ncols=len(results))
    fig.suptitle("Success rate under visual perturbations (shaded: 95% CI)", x=0.01, ha="left",
                 fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig
