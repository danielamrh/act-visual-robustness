"""Figures for the feature probe (`avr.analysis.features`)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from avr.analysis.robustness import FACTOR_ORDER, GRID, SERIES_COLORS, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

ATTN_RGB = np.array([0x2A, 0x78, 0xD6]) / 255  # sequential blue (reference palette, step 450)

LABELS = {
    "None_L0": "clean",
    "shadows_off_L1": "shadows off",
    "light_direction_L2": "light rotated 30°",
    "noise_L3": "noise σ=16",
    "blur_L3": "blur r=2.5",
    "cube_color_L4": "cube recolored",
}


def load_figure_maps(npz_path) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """{condition: {step: {"img": HxWx3, "attn": hxw}}} from a figures_seed*.npz."""
    data = np.load(npz_path)
    out: dict = defaultdict(dict)
    for key in data.files:
        name, step, kind = key.split("__")
        out[name].setdefault(int(step.removeprefix("step")), {})[kind] = data[key]
    out = dict(out)
    if "clean" in out:  # stored under "clean" by analyze_seed
        out["None_L0"] = out.pop("clean")
    return out


def overlay(img: np.ndarray, attn: np.ndarray, vmax: float, max_alpha: float = 0.85, gamma: float = 0.5) -> np.ndarray:
    """Blend the map (upsampled to the image) as blue with opacity ~ (value / vmax) ** gamma.
    The square-root scale keeps weaker structure visible next to a few very hot patches."""
    from PIL import Image

    h, w = img.shape[:2]
    a = np.asarray(Image.fromarray(attn.astype(np.float32)).resize((w, h), Image.BILINEAR))
    a = np.clip(a / vmax, 0, 1)[..., None] ** gamma * max_alpha
    return (img / 255 * (1 - a) + ATTN_RGB * a).clip(0, 1)


def attention_figure(maps: dict, step: int, kind: str = "occl", title: str = "", path=None):
    """One column per condition: camera image (top) and a sensitivity overlay (bottom), shared scale.

    kind="occl": occlusion sensitivity (how much the action chunk changes when a patch is hidden);
    kind="attn": decoder cross-attention. On the clean image the shadow mask is outlined in orange.
    """
    import matplotlib.pyplot as plt

    names = [n for n in LABELS if n in maps and step in maps[n] and kind in maps[n][step]]
    vmax = max(maps[n][step][kind].max() for n in names)
    fig, axes = plt.subplots(2, len(names), figsize=(2.6 * len(names), 4.3), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for j, n in enumerate(names):
        entry = maps[n][step]
        axes[0, j].imshow(entry["img"])
        axes[1, j].imshow(overlay(entry["img"], entry[kind], vmax))
        if "shadow_mask" in entry:
            for ax in axes[:, j]:
                ax.contour(entry["shadow_mask"], levels=[0.5], colors=[SERIES_COLORS[1]], linewidths=0.9)
        axes[0, j].set_title(LABELS[n], fontsize=9.5, color=TEXT_PRIMARY)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    what = {"occl": "occlusion sensitivity", "attn": "cross-attention"}[kind]
    axes[0, 0].set_ylabel("camera", fontsize=9, color=TEXT_SECONDARY)
    axes[1, 0].set_ylabel(what, fontsize=9, color=TEXT_SECONDARY)
    default = f"Where the policy looks: {what} (step {step}, shared sqrt scale; orange = shadows)"
    fig.suptitle(title or default, x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def load_probe_rows(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def cell_means(rows: list[dict], metric: str) -> dict[tuple[str, int], tuple[float, float]]:
    """{(factor, level): (mean, standard error over seeds)}; probe steps averaged per seed first."""
    per_seed: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        per_seed[(r["factor"], r["level"])][r["seed"]].append(r[metric])
    out = {}
    for cell, seeds in per_seed.items():
        vals = np.array([np.mean(v) for v in seeds.values()])
        out[cell] = (float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
    return out


def plot_shift_vs_drop(policies: dict[str, tuple[list[dict], list[dict]]], metric="action_shift", path=None):
    """Scatter per policy: probe shift of each cell (x) vs its paired success drop (y).

    `policies` maps a label to (probe rows, robustness summary rows with a `delta` column).
    Returns the figure and {label: Spearman rho}.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rhos = {}
    for i, (label, (probe_rows, summary)) in enumerate(policies.items()):
        means = cell_means(probe_rows, metric)
        pts = [(means[(r["factor"], r["level"])][0], r["delta"]) for r in summary
               if (r["factor"], r["level"]) in means and "delta" in r]  # fmt: skip
        xs, ys = zip(*pts)
        rhos[label] = float(spearmanr(xs, ys).statistic)
        ax.scatter(xs, ys, s=34, color=SERIES_COLORS[i], edgecolor=SURFACE, linewidth=1.5,
                   label=f"{label} (Spearman ρ = {rhos[label]:.2f})")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xlabel(metric.replace("_", " ") + " (clean vs. perturbed image, same state)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel("success change vs. clean (paired)", fontsize=9, color=TEXT_SECONDARY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Does the probe predict the damage? One point per factor × level", fontsize=10.5,
                 color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig, rhos


def shift_table(rows: list[dict]) -> list[dict]:
    """Mean ± s.e. of the three probe metrics per cell, in factor order."""
    metrics = ("feature_shift", "action_shift", "attn_shift")
    means = {m: cell_means(rows, m) for m in metrics}
    cells = sorted(means["feature_shift"], key=lambda c: (FACTOR_ORDER.index(c[0]) if c[0] in FACTOR_ORDER
                                                          else len(FACTOR_ORDER), c[1]))  # fmt: skip
    return [{"factor": f, "level": lv, **{m: means[m][(f, lv)][0] for m in metrics},
             **{f"{m}_se": means[m][(f, lv)][1] for m in metrics}} for f, lv in cells]  # fmt: skip
