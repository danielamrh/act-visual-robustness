"""Grid of rendered first frames: one row per factor, one column per level."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from avr.perturb.suite import FACTORS, LEVELS


def preview_grid(seed: int = 0, factors=None, levels=LEVELS, scale: float = 0.3, task="transfer_cube") -> Image.Image:
    from avr.perturb.env import PerturbedAlohaEnv

    factors = list(factors or FACTORS)
    env = PerturbedAlohaEnv(task=task)
    tiles = []
    for name in factors:
        row = []
        for level in levels:
            env.set_perturbation(name, level)
            obs, _ = env.reset(seed=seed)
            row.append(Image.fromarray(obs["pixels"]["top"]))
        tiles.append(row)
    env.close()

    w, h = (int(d * scale) for d in tiles[0][0].size)
    label_w, header_h = 130, 22
    grid = Image.new("RGB", (label_w + w * len(levels), header_h + h * len(factors)), "white")
    draw = ImageDraw.Draw(grid)
    for j, level in enumerate(levels):
        draw.text((label_w + j * w + w // 2 - 20, 5), f"level {level}", fill="black")
    for i, name in enumerate(factors):
        draw.text((5, header_h + i * h + h // 2 - 6), name, fill="black")
        for j, tile in enumerate(tiles[i]):
            grid.paste(tile.resize((w, h)), (label_w + j * w, header_h + i * h))
    return grid


def as_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img)
