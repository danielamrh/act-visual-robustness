"""Shadow augmentation: remove or move the shadows cast on the table.

The analysis showed that the baseline relies on the shadows of grippers and cube (switching them off halves its
success) and that standard image augmentation makes this *worse*: color / brightness / contrast jitter leaves
shadows intact. This transform edits the shadows themselves, on the recorded training images:

  remove: shadow pixels on the table are divided by the shadow factor (like switching shadows off)
  move:   shadows are removed and re-drawn shifted and stretched: one random direction per image, along which every
          shadow gets longer or shorter (and a bit wider or narrower) around its own center, as a different light
          position would do; a cheap image-space stand-in for changing the light, not a physical re-rendering

Detection exploits the simple scene: the table is a uniform, unsaturated grey. Its lit level L is estimated per
image (robust to the brightness / contrast jitter applied before), shadows are grey pixels darker than 0.93 L inside
the table's bounding box, and small dark components (shading on the arms) are dropped. How well this matches the true
shadows is measured in simulation, where the exact shadow mask is known (`avr.analysis.shadow_validation`).

Registered as LeRobot image transform "ShadowEdit" by the act_enc plugin, so it can be added to
`--dataset.image_transforms.tfs` next to LeRobot's built-in transforms.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import Transform

GREY_MAX_SAT = 0.25  # table, background and shadows are unsaturated
# Measured in simulation (avr.analysis.shadow_validation, true shadow masks from renders with / without shadows):
# only one light casts shadows, so a shadow multiplies the table's color by ~0.83 (0.78-0.88), equally in all
# channels. The lit table has two regions (brightness ~0.29 and ~0.33), so the brightest shadow (~0.27, in the bright
# region) is still darker than the darkest lit table (~0.275): with the *darker* lit level L_lo as reference, a
# threshold of 0.93 L_lo separates shadow from lit table in both regions.
SHADOW_FACTOR = 0.83
SHADOW_BELOW = 0.93  # shadow: v < 0.93 L_lo
SHADOW_ABOVE = 0.60  # ...and v > 0.60 L_lo (darker is the black background or dark arm parts)
LIT_RANGE = (0.95, 1.30)  # lit table: L_lo * 0.95 ... L_hi * 1.30
MIN_COMPONENT = 150  # px; smaller dark grey blobs are shading on the arms, not cast shadows


def _value_saturation(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    v, _ = img.max(0)
    mn, _ = img.min(0)
    return v, (v - mn) / (v + 1e-6)


def _densest_level(cand: torch.Tensor, lo: float, hi: float) -> tuple[float, int]:
    """Center of the ±4 % brightness window in [lo, hi] holding the most values, and that count."""
    centers = torch.linspace(lo, hi, 341)
    counts = torch.searchsorted(cand, centers * 1.04) - torch.searchsorted(cand, centers * 0.96)
    k = int(counts.argmax())
    best = float(centers[k])
    near = cand[(cand > 0.96 * best) & (cand < 1.04 * best)]
    return float(near.median()) if near.numel() else best, int(counts[k])


def table_levels(img: torch.Tensor) -> tuple[float, float]:
    """Brightness of the lit table (L_lo, L_hi); equal if only one lit region is found.

    Counting pixels per relative brightness window (not a plain histogram mode) matters: shadow pixels all share
    one exact value, while a lit region spreads over several close values.
    """
    v, s = _value_saturation(img[..., ::2, ::2])  # a level is a statistic: every 4th pixel is plenty (speed)
    cand = v[(s < GREY_MAX_SAT) & (v > 0.05) & (v < 0.9)]
    if cand.numel() < 100:
        return float("nan"), float("nan")
    cand, _ = cand.sort()
    l1, n1 = _densest_level(cand, 0.05, 0.9)
    # a second lit region: brighter than the first one's shadows, not overlapping the first window
    rest = cand[(cand < 0.92 * l1) | (cand > 1.08 * l1)]
    rest = rest[(rest > 0.88 * l1) & (rest < 1.25 * l1)]
    if rest.numel() > 0.2 * n1:
        l2, n2 = _densest_level(rest, 0.88 * l1, 1.25 * l1)
        if n2 > 0.2 * n1:
            return min(l1, l2), max(l1, l2)
    return l1, l1


def table_level(img: torch.Tensor) -> float:
    """The darker lit table level (reference for shadow detection)."""
    return table_levels(img)[0]


def _drop_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    from scipy import ndimage

    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_size
    return keep[labels]


def _quantile_span(counts: torch.Tensor, q: float = 0.005) -> tuple[int, int]:
    """Indices of the q and 1-q quantiles of a distribution given as counts per index."""
    cdf = counts.cumsum(0).float() / counts.sum()
    lo = int(torch.searchsorted(cdf, torch.tensor(q)))
    hi = int(torch.searchsorted(cdf, torch.tensor(1 - q)))
    return lo, min(hi, len(counts) - 1)


def detect_shadows(img: torch.Tensor) -> dict:
    """Shadow and lit-table masks of a (3, H, W) image in [0, 1].

    Returns {"shadow": bool (H, W), "lit": bool (H, W), "level": L_lo, "factor": shadow darkening factor}.
    """
    v, s = _value_saturation(img)
    l_lo, l_hi = table_levels(img)
    h, w = v.shape
    empty = torch.zeros(h, w, dtype=torch.bool)
    if not np.isfinite(l_lo):
        return {"shadow": empty, "lit": empty, "level": l_lo, "factor": SHADOW_FACTOR}
    grey = s < GREY_MAX_SAT
    lit = grey & (v > LIT_RANGE[0] * l_lo) & (v < LIT_RANGE[1] * l_hi)

    # table region: robust bounding box of the lit table pixels (both lit regions)
    if lit.sum() < 500:
        return {"shadow": empty, "lit": lit, "level": l_lo, "factor": SHADOW_FACTOR}
    y0, y1 = _quantile_span(lit.sum(1))
    x0, x1 = _quantile_span(lit.sum(0))
    in_table = torch.zeros(h, w, dtype=torch.bool)
    in_table[y0 : y1 + 1, x0 : x1 + 1] = True

    dark = grey & in_table & (v > SHADOW_ABOVE * l_lo) & (v < SHADOW_BELOW * l_lo)
    shadow = torch.from_numpy(_drop_small_components(dark.numpy(), MIN_COMPONENT))
    return {"shadow": shadow, "lit": lit, "level": l_lo, "factor": SHADOW_FACTOR}


def _soften(mask: torch.Tensor, k: int = 5) -> torch.Tensor:
    m = mask.float()[None, None]
    return F.avg_pool2d(m, k, stride=1, padding=k // 2)[0, 0].clamp(0, 1)


def _dilate(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    return F.max_pool2d(mask.float()[None, None], k, stride=1, padding=k // 2)[0, 0] > 0


def remove_shadows(img: torch.Tensor, det: dict, strength: float = 1.0) -> torch.Tensor:
    """Undo the shadow darkening: divide shadow pixels by the shadow factor (strength 1 = fully removed).
    Dividing (rather than painting one table color) restores each lit region's own brightness."""
    if not det["shadow"].any():
        return img
    # hard mask: softening or dilating the edge lifts lit pixels next to the shadow and was measurably worse
    # (removed 0.68 vs 0.59-0.65 in simulation); the rest is the penumbra and the spread of the factor
    gain = 1 + (1 / det["factor"] - 1) * det["shadow"].float() * strength
    return (img * gain).clamp(0, 1)


def warp_mask(mask: torch.Tensor, dy: float, dx: float, angle: float = 0.0, length: float = 1.0,
              width: float = 1.0) -> torch.Tensor:
    """Each connected shadow scaled around its own center (by `length` along `angle`, by `width` across it), then
    shifted by (dy, dx). One direction for all shadows in the image, as with a single light."""
    from scipy import ndimage

    labels, n = ndimage.label(mask.numpy())
    if n == 0:
        return mask
    h, w = mask.shape
    c, s = math.cos(angle), math.sin(angle)
    # forward map p' = center + t + M (p - center) on (x, y), M = R diag(length, width) R^T, axis = (cos, sin)
    m = np.array([[c * c * length + s * s * width, c * s * (length - width)],
                  [c * s * (length - width), s * s * length + c * c * width]])
    m_inv = np.linalg.inv(m)
    out = np.zeros((h, w), dtype=bool)
    for k, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = np.nonzero(labels[sl] == k)
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start
        # target box: forward-mapped corners of the component's box
        corners = np.array([[sl[1].start, sl[0].start], [sl[1].stop, sl[0].start],
                            [sl[1].start, sl[0].stop], [sl[1].stop, sl[0].stop]], dtype=float) - (cx, cy)
        tgt = corners @ m.T + (cx + dx, cy + dy)
        x0, y0 = np.floor(tgt.min(0)).astype(int) - 1
        x1, y1 = np.ceil(tgt.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        src = np.stack([xx - cx - dx, yy - cy - dy], -1) @ m_inv.T + (cx, cy)  # nearest source pixel
        sx, sy = np.rint(src[..., 0]).astype(int), np.rint(src[..., 1]).astype(int)
        ok = (sx >= 0) & (sx < w) & (sy >= 0) & (sy < h)
        hit = np.zeros_like(ok)
        hit[ok] = labels[sy[ok], sx[ok]] == k
        out[y0:y1, x0:x1] |= hit
    return torch.from_numpy(out)


def shift_shadows(img: torch.Tensor, det: dict, dy: int, dx: int, strength: float = 1.0, angle: float = 0.0,
                  length: float = 1.0, width: float = 1.0) -> torch.Tensor:
    """Remove the shadows, then cast them again at the mask moved by (dy, dx) and stretched (`warp_mask`), on lit
    table pixels only (never on the arms, the cube or the background)."""
    out = remove_shadows(img, det, strength)
    if not det["shadow"].any():
        return out
    moved = warp_mask(det["shadow"], dy, dx, angle, length, width)
    table = det["lit"] | det["shadow"]
    m = _soften(moved & table) * strength
    return (out * (1 - m * (1 - det["factor"]))).clamp(0, 1)


class ShadowEdit(Transform):
    """LeRobot / torchvision v2 transform: remove (p_remove) or move the table shadows.

    Args:
        p: probability of editing a frame at all.
        p_remove: probability of removing instead of moving the shadows (if edited).
        max_shift: maximum shift as a fraction of the image size.
        length: (min, max) scale of the shadows along a random direction.
        width: (min, max) scale across that direction.
        strength: (min, max) of the edit strength, 1 = complete.

    The stretch ranges are plausible changes of the light position, chosen without looking at the test suite's
    light-direction levels (which would tune the augmentation to the test).
    """

    def __init__(self, p: float = 0.5, p_remove: float = 0.5, max_shift: float = 0.08, length=(0.7, 1.5),
                 width=(0.85, 1.15), strength=(0.6, 1.0)) -> None:
        super().__init__()
        self.p = float(p)
        self.length = (float(length[0]), float(length[1]))
        self.width = (float(width[0]), float(width[1]))
        self.p_remove = float(p_remove)
        self.max_shift = float(max_shift)
        self.strength = (float(strength[0]), float(strength[1]))

    def make_params(self, flat_inputs: list) -> dict:
        return {
            "apply": torch.rand(1).item() < self.p,
            "remove": torch.rand(1).item() < self.p_remove,
            "shift": (torch.rand(2) * 2 - 1).tolist(),
            "strength": torch.empty(1).uniform_(*self.strength).item(),
            "angle": torch.rand(1).item() * math.pi,
            "length": torch.empty(1).uniform_(*self.length).item(),
            "width": torch.empty(1).uniform_(*self.width).item(),
        }

    def transform(self, inpt, params: dict):
        if not params["apply"] or not isinstance(inpt, torch.Tensor) or not inpt.is_floating_point():
            return inpt
        if inpt.ndim < 3 or inpt.shape[-3] != 3:
            return inpt
        if inpt.ndim > 3:  # (..., 3, H, W): edit each frame on its own
            flat = inpt.reshape(-1, *inpt.shape[-3:])
            return torch.stack([self.transform(f, params) for f in flat]).reshape(inpt.shape)
        det = detect_shadows(inpt)
        if params["remove"]:
            return remove_shadows(inpt, det, params["strength"])
        h, w = inpt.shape[-2:]
        dy, dx = (round(params["shift"][0] * self.max_shift * h), round(params["shift"][1] * self.max_shift * w))
        return shift_shadows(inpt, det, dy, dx, params["strength"], params["angle"], params["length"],
                             params["width"])


# Large enough that LeRobot's RandomSubsetApply (weighted sampling without replacement) always picks ShadowEdit, so
# with max_num_transforms = 4 the other three are drawn exactly as in run A; ShadowEdit runs last (dict order), after
# the color jitter, as in the validation. Whether a frame is edited is decided by ShadowEdit's own `p`.
ALWAYS = 1e6


def lerobot_flags(**shadow_kwargs) -> list[str]:
    """lerobot-train flags for run B: LeRobot's default image augmentation (run A) plus ShadowEdit.

    draccus can only replace `tfs` as a whole (JSON), so the defaults are copied from LeRobot's config."""
    import dataclasses
    import json

    from lerobot.transforms.transforms import ImageTransformsConfig

    tfs = {name: dataclasses.asdict(cfg) for name, cfg in ImageTransformsConfig().tfs.items()}
    tfs["shadow"] = {"weight": ALWAYS, "type": "ShadowEdit", "kwargs": shadow_kwargs}
    return [
        "--dataset.image_transforms.enable=true",
        f"--dataset.image_transforms.max_num_transforms={ImageTransformsConfig().max_num_transforms + 1}",
        f"--dataset.image_transforms.tfs={json.dumps(tfs)}",
    ]


def register() -> None:
    """Make "ShadowEdit" available as a LeRobot image transform type."""
    from lerobot.transforms import transforms

    transforms._CUSTOM_TRANSFORMS.setdefault("ShadowEdit", ShadowEdit)
