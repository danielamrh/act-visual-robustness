"""Visual perturbation factors with 5 severity levels (0 = none ... 4 = strong).

Scene factors edit the MuJoCo model before an episode starts (lights, colors,
background, distractors, camera); image factors corrupt each rendered frame.
None of them changes physics: the robot and cube behave identically, only
what the policy *sees* differs.

`category` separates nuisances a good policy should ignore from changes to
the task-relevant object itself (the cube's color).

The level magnitudes are a first calibration; check them visually with
`avr.perturb.preview` before running experiments.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable

import numpy as np

from avr.perturb import scene

LEVELS = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class Factor:
    name: str
    kind: str  # "scene" | "image"
    category: str  # "nuisance" | "task"
    description: str


FACTORS = {
    f.name: f
    for f in [
        Factor("light_intensity", "scene", "nuisance", "all lights dimmer or brighter"),
        Factor("light_color", "scene", "nuisance", "colored tint of all lights"),
        Factor("light_direction", "scene", "nuisance", "directional lights rotated"),
        Factor("table_color", "scene", "nuisance", "table recolored"),
        Factor("background", "scene", "nuisance", "floor behind the table: flat colors, then checker textures"),
        Factor("distractors", "scene", "nuisance", "level = number of extra objects on the table"),
        Factor("camera_pose", "scene", "nuisance", "top camera moved and zoomed"),
        Factor("noise", "image", "nuisance", "Gaussian pixel noise"),
        Factor("blur", "image", "nuisance", "Gaussian blur"),
        Factor("jpeg", "image", "nuisance", "JPEG compression artifacts"),
        Factor("cube_color", "scene", "task", "the cube itself recolored"),
    ]
}


def _pick(values, level):
    return values[level]


# ---------------------------------------------------------------- model state


class ModelState:
    """Snapshot of every model field a scene factor may touch."""

    FIELDS = (
        "light_diffuse", "light_specular", "light_ambient", "light_dir",
        "geom_rgba", "geom_group", "geom_size", "geom_matid",
        "body_pos", "cam_pos", "cam_quat", "cam_fovy", "cam_mode", "cam_bodyid",
    )  # fmt: skip
    HEADLIGHT = ("ambient", "diffuse", "specular")

    def __init__(self, physics):
        m = physics.model
        self.arrays = {f: np.array(getattr(m, f), copy=True) for f in self.FIELDS}
        self.headlight = {k: np.array(getattr(m.vis.headlight, k), copy=True) for k in self.HEADLIGHT}

    def restore(self, physics):
        m = physics.model
        for f, value in self.arrays.items():
            getattr(m, f)[...] = value
        for k, value in self.headlight.items():
            getattr(m.vis.headlight, k)[...] = value


# ---------------------------------------------------------------- scene factors


def _scale_lights(m, factor):
    """Multiply light colors by `factor` (scalar or RGB), clipped to [0, 1]."""
    for f in ("light_diffuse", "light_specular"):
        arr = getattr(m, f)
        arr[...] = np.clip(arr * factor, 0, 1)
    for k in ("ambient", "diffuse", "specular"):
        arr = getattr(m.vis.headlight, k)
        arr[...] = np.clip(arr * factor, 0, 1)


def light_intensity(physics, rng, level, **_):
    u = _pick([0.0, 0.3, 0.6, 0.9, 1.2], level)
    scale = float(np.exp(u * rng.choice([-1.0, 1.0])))
    _scale_lights(physics.model, scale)
    return {"scale": scale}


def light_color(physics, rng, level, **_):
    s = _pick([0.0, 0.2, 0.4, 0.6, 0.8], level)
    d = rng.normal(size=3)
    d -= d.mean()
    d /= np.abs(d).max() + 1e-8
    tint = 1.0 + s * d
    _scale_lights(physics.model, tint)
    return {"tint": tint.round(3).tolist()}


def _rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def light_direction(physics, rng, level, **_):
    deg = _pick([0, 15, 30, 45, 60], level)
    rot = _rotation(rng.normal(size=3), np.deg2rad(deg))
    m = physics.model
    m.light_dir[...] = m.light_dir @ rot.T
    return {"degrees": deg}


def table_color(physics, rng, level, **_):
    a = level / 4
    target = scene.random_color(rng, avoid_red=True, val=(0.15, 0.85))
    rgba = physics.named.model.geom_rgba["table"]
    rgba[:3] = (1 - a) * rgba[:3] + a * target
    return {"rgb": rgba[:3].round(3).tolist()}


def background(physics, rng, level, **_):
    m = physics.model
    gid = m.name2id(scene.FLOOR_GEOM, "geom")
    m.geom_group[gid] = scene.VISIBLE_GROUP
    if level <= 2:
        val = (0.2, 0.35) if level == 1 else (0.6, 0.9)
        color = scene.random_color(rng, avoid_red=True, sat=(0.3, 0.8), val=val)
        m.geom_matid[gid] = -1
        m.geom_rgba[gid] = [*color, 1.0]
        return {"floor": "flat", "rgb": color.round(3).tolist()}
    contrast = "low" if level == 3 else "high"
    name = scene.checker_material(contrast, int(rng.integers(scene.N_CHECKERS)))
    m.geom_matid[gid] = m.name2id(name, "material")
    m.geom_rgba[gid] = [1.0, 1.0, 1.0, 1.0]
    return {"floor": name}


def distractors(physics, rng, level, avoid_xy=(), **_):
    m = physics.model
    placed = []
    params = []
    for slot in range(level):
        shape = scene.DISTRACTOR_SHAPES[rng.integers(len(scene.DISTRACTOR_SHAPES))]
        size = rng.uniform(0.02, 0.035)
        color = scene.random_color(rng, avoid_red=True)
        for _ in range(1000):
            xy = rng.uniform([scene.TABLE_X[0], scene.TABLE_Y[0]], [scene.TABLE_X[1], scene.TABLE_Y[1]])
            if all(np.linalg.norm(xy - np.asarray(a)) > 0.15 for a in avoid_xy) and all(
                np.linalg.norm(xy - p) > 0.09 for p in placed
            ):
                break
        placed.append(xy)
        gid = m.name2id(scene.distractor_geom(slot, shape), "geom")
        m.geom_group[gid] = scene.VISIBLE_GROUP
        m.geom_rgba[gid] = [*color, 1.0]
        m.geom_size[gid] = {"box": [size, size, size], "sphere": [size, 0, 0], "cylinder": [size, size, 0]}[shape]
        m.body_pos[m.name2id(scene.distractor_body(slot), "body")] = [xy[0], xy[1], size]
        params.append({"shape": shape, "xy": xy.round(3).tolist(), "size": round(size, 3)})
    return {"objects": params}


def camera_pose(physics, rng, level, **_):
    """Shift, tilt and zoom the top camera.

    The top camera looks straight down at its target body (mode
    `targetbody`), where the image "up" direction is degenerate: moving it a
    few cm rotates the image by tens of degrees. So we switch it to a fixed
    camera with its current world orientation and apply small, controlled
    changes on top.
    """
    import mujoco

    dist = _pick([0.0, 0.02, 0.04, 0.07, 0.10], level)
    tilt = _pick([0, 2, 4, 6, 8], level)
    dfov = _pick([0, 2, 4, 6, 8], level) * rng.choice([-1, 1])

    m, d = physics.model, physics.data
    cam = m.name2id("top", "camera")
    mujoco.mj_kinematics(m.ptr, d.ptr)
    mujoco.mj_camlight(m.ptr, d.ptr)
    xmat = d.cam_xmat[cam].reshape(3, 3)
    xpos = d.cam_xpos[cam].copy()

    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    rot = _rotation(rng.normal(size=3), np.deg2rad(tilt)) @ xmat
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())

    m.cam_mode[cam] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    m.cam_bodyid[cam] = 0  # fixed in the world frame
    m.cam_pos[cam] = xpos + dist * direction
    m.cam_quat[cam] = quat
    m.cam_fovy[cam] += dfov
    return {"offset_m": (dist * direction).round(3).tolist(), "tilt_deg": tilt, "dfovy": float(dfov)}


def cube_color(physics, rng, level, **_):
    a = level / 4
    target = scene.random_color(rng, avoid_red=True)
    rgba = physics.named.model.geom_rgba["red_box"]
    rgba[:3] = (1 - a) * rgba[:3] + a * target
    return {"rgb": rgba[:3].round(3).tolist()}


SCENE_FNS: dict[str, Callable] = {
    "light_intensity": light_intensity,
    "light_color": light_color,
    "light_direction": light_direction,
    "table_color": table_color,
    "background": background,
    "distractors": distractors,
    "camera_pose": camera_pose,
    "cube_color": cube_color,
}


# ---------------------------------------------------------------- image factors


def make_image_fn(name: str, level: int, rng: np.random.Generator) -> tuple[Callable, dict]:
    """Per-episode image corruption `img (H,W,3 uint8) -> img`."""
    if name == "noise":
        sigma = _pick([0, 4, 8, 16, 32], level)

        def fn(img):
            noisy = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
            return np.clip(noisy, 0, 255).astype(np.uint8)

        return fn, {"sigma": sigma}
    if name == "blur":
        from PIL import Image, ImageFilter

        radius = _pick([0, 0.75, 1.5, 2.5, 3.5], level)
        return (lambda img: np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius)))), {"radius": radius}
    if name == "jpeg":
        from PIL import Image

        quality = _pick([100, 40, 20, 10, 5], level)

        def fn(img):
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            return np.asarray(Image.open(buf).convert("RGB"))

        return fn, {"quality": quality}
    raise KeyError(name)


def apply_scene_factor(name: str, level: int, physics, rng: np.random.Generator, **kwargs) -> dict:
    return SCENE_FNS[name](physics, rng, level, **kwargs)
