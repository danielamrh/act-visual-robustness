"""Augmented ALOHA scene: the original gym-aloha model plus hidden, purely
visual extras that the perturbation suite can switch on at runtime.

Extras (all `contype=0 conaffinity=0`, so they never collide or touch, and
static, so they add no degrees of freedom -> physics and qpos layout are
identical to the original task):
  - a floor plane below the table (background), with flat and checker materials
  - distractor objects on the table (box / sphere / cylinder per slot)

Hidden geoms live in geom group 3, which MuJoCo does not render by default;
showing one means moving it to group 0.
"""

from __future__ import annotations

import colorsys
import shutil
import tempfile
from pathlib import Path

import numpy as np

HIDDEN_GROUP = 3
VISIBLE_GROUP = 0

N_DISTRACTORS = 4
DISTRACTOR_SHAPES = ("box", "sphere", "cylinder")
N_CHECKERS = 4  # checker materials per contrast level

TASK_XML = {
    "transfer_cube": "bimanual_viperx_transfer_cube.xml",
    "insertion": "bimanual_viperx_insertion.xml",
}

# Table top is at z = 0; the table spans x in [-0.61, 0.61], y in [0.22, 0.98].
TABLE_X = (-0.55, 0.55)
TABLE_Y = (0.28, 0.92)
FLOOR_Z = -0.75


def distractor_geom(slot: int, shape: str) -> str:
    return f"avr_distractor{slot}_{shape}"


def distractor_body(slot: int) -> str:
    return f"avr_distractor{slot}"


FLOOR_GEOM = "avr_floor"


def checker_material(contrast: str, i: int) -> str:
    return f"avr_checker_{contrast}{i}"


def _checker_colors(rng: np.random.Generator, contrast: str) -> tuple[str, str]:
    hue = rng.uniform()
    if contrast == "low":
        v1, v2, sat = 0.35, 0.5, 0.35
    else:
        v1, v2, sat = 0.1, 0.9, 0.7
    c1 = colorsys.hsv_to_rgb(hue, sat, v1)
    c2 = colorsys.hsv_to_rgb((hue + 0.5) % 1.0, sat, v2)
    fmt = lambda c: " ".join(f"{x:.3f}" for x in c)  # noqa: E731
    return fmt(c1), fmt(c2)


def _extras_xml() -> tuple[str, str]:
    """(asset xml, worldbody xml) for the extras. Deterministic."""
    rng = np.random.default_rng(231)
    assets = []
    for contrast in ("low", "high"):
        for i in range(N_CHECKERS):
            c1, c2 = _checker_colors(rng, contrast)
            name = checker_material(contrast, i)
            assets.append(
                f'<texture name="{name}_tex" type="2d" builtin="checker" width="256" height="256" '
                f'rgb1="{c1}" rgb2="{c2}"/>'
                f'<material name="{name}" texture="{name}_tex" texrepeat="{6 + 2 * i} {6 + 2 * i}"/>'
            )
    visual_only = f'contype="0" conaffinity="0" group="{HIDDEN_GROUP}"'
    bodies = [
        f'<body name="avr_floor_body" pos="0 0.6 {FLOOR_Z}">'
        f'<geom name="{FLOOR_GEOM}" type="plane" size="4 4 0.1" rgba="0.5 0.5 0.5 1" {visual_only}/>'
        f"</body>"
    ]
    for slot in range(N_DISTRACTORS):
        geoms = [
            f'<geom name="{distractor_geom(slot, "box")}" type="box" size="0.025 0.025 0.025" {visual_only}/>',
            f'<geom name="{distractor_geom(slot, "sphere")}" type="sphere" size="0.025" {visual_only}/>',
            f'<geom name="{distractor_geom(slot, "cylinder")}" type="cylinder" size="0.025 0.025" {visual_only}/>',
        ]
        bodies.append(f'<body name="{distractor_body(slot)}" pos="0 0.6 -1">{"".join(geoms)}</body>')
    return "".join(assets), "".join(bodies)


_CACHE: dict[str, Path] = {}


def augmented_xml_path(task: str = "transfer_cube") -> Path:
    """Write (once per process) a copy of the gym-aloha assets with the
    augmented task XML and return its path."""
    if task in _CACHE:
        return _CACHE[task]
    from gym_aloha.constants import ASSETS_DIR

    import mujoco

    out_dir = Path(tempfile.mkdtemp(prefix="avr_aloha_assets_"))
    shutil.copytree(ASSETS_DIR, out_dir, dirs_exist_ok=True)
    original = Path(ASSETS_DIR) / TASK_XML[task]
    xml = original.read_text()
    # The extras enlarge the scene's bounding statistics, which MuJoCo uses for
    # the shadow-map frustum and clipping planes -> slightly different shadows.
    # Pin them to the original model's values so unperturbed frames are identical.
    stat = mujoco.MjModel.from_xml_path(str(original)).stat
    statistic = (
        f'<statistic extent="{stat.extent!r}" '
        f'center="{" ".join(repr(float(c)) for c in stat.center)}"/>'
    )
    assets, bodies = _extras_xml()
    assert xml.count("<worldbody>") == 1
    xml = xml.replace("<worldbody>", f"{statistic}<asset>{assets}</asset>\n    <worldbody>{bodies}", 1)
    path = out_dir / f"avr_{TASK_XML[task]}"
    path.write_text(xml)
    _CACHE[task] = path
    return path


def random_color(rng: np.random.Generator, avoid_red: bool = True, sat=(0.5, 1.0), val=(0.3, 0.9)) -> np.ndarray:
    """Random saturated RGB color. `avoid_red` keeps hues away from the red cube."""
    while True:
        hue = rng.uniform()
        if not avoid_red or 0.1 < hue < 0.9:
            break
    return np.array(colorsys.hsv_to_rgb(hue, rng.uniform(*sat), rng.uniform(*val)))
