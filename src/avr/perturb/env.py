"""gym-aloha env with a perturbation factor applied per episode.

Drop-in for `gym_aloha.env.AlohaEnv` (same observations and actions), with
two differences:
  - the scene is the augmented one from `avr.perturb.scene`
  - only the `top` camera is rendered per step (gym-aloha also renders
    `angle` and `front_close` every step and discards them)

Perturbation parameters are drawn from an RNG seeded by (episode seed,
factor, level), so a given episode sees the same perturbation no matter which
policy is evaluated: comparisons between encoders are paired.
"""

from __future__ import annotations

import collections
import zlib

import numpy as np
from dm_control import mujoco
from dm_control.rl import control
from gym_aloha.constants import DT
from gym_aloha.env import AlohaEnv
from gym_aloha.tasks.sim import InsertionTask, TransferCubeTask
from gym_aloha.utils import sample_box_pose

from avr.perturb import scene
from avr.perturb.suite import FACTORS, ModelState, apply_scene_factor, make_image_fn


def _top_only_observation(task, physics):
    obs = collections.OrderedDict()
    obs["qpos"] = task.get_qpos(physics)
    obs["qvel"] = task.get_qvel(physics)
    obs["env_state"] = task.get_env_state(physics)
    obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")}
    return obs


class TopCameraTransferCubeTask(TransferCubeTask):
    def get_observation(self, physics):
        return _top_only_observation(self, physics)


class TopCameraInsertionTask(InsertionTask):
    def get_observation(self, physics):
        return _top_only_observation(self, physics)


TASKS = {"transfer_cube": TopCameraTransferCubeTask, "insertion": TopCameraInsertionTask}


class PerturbedAlohaEnv(AlohaEnv):
    def __init__(self, task="transfer_cube", factor: str | None = None, level: int = 0,
                 obs_type="pixels_agent_pos", **kwargs):
        if factor is not None and factor not in FACTORS:
            raise KeyError(f"unknown factor {factor!r}; choose from {sorted(FACTORS)}")
        self.factor = factor
        self.level = level
        self.episode_params: dict = {}
        self._image_fn = None
        super().__init__(task=task, obs_type=obs_type, **kwargs)
        self._default_state = ModelState(self._env.physics)

    def _make_env_task(self, task_name):
        physics = mujoco.Physics.from_xml_path(str(scene.augmented_xml_path(task_name)))
        return control.Environment(
            physics, TASKS[task_name](), float("inf"), control_timestep=DT, n_sub_steps=None, flat_observation=False
        )

    def set_perturbation(self, factor: str | None, level: int = 0):
        """Takes effect at the next reset."""
        if factor is not None and factor not in FACTORS:
            raise KeyError(factor)
        self.factor, self.level = factor, level

    def _avoid_xy(self, seed):
        if self.task == "transfer_cube":
            return [sample_box_pose(seed)[:2]]  # the pose AlohaEnv.reset will use
        return []

    def reset(self, seed=None, options=None):
        physics = self._env.physics
        self._default_state.restore(physics)
        self._image_fn = None
        self.episode_params = {"factor": self.factor, "level": self.level}
        if self.factor is not None and self.level > 0:
            rng = np.random.default_rng(
                [seed if seed is not None else np.random.randint(2**31), zlib.crc32(self.factor.encode()), self.level]
            )
            if FACTORS[self.factor].kind == "scene":
                params = apply_scene_factor(self.factor, self.level, physics, rng, avoid_xy=self._avoid_xy(seed))
            else:
                self._image_fn, params = make_image_fn(self.factor, self.level, rng)
            self.episode_params["params"] = params
        obs, info = super().reset(seed=seed, options=options)  # renders with the edited model
        info["perturbation"] = self.episode_params
        return obs, info

    def _corrupt(self, img):
        return self._image_fn(img) if self._image_fn is not None else img

    def _format_raw_obs(self, raw_obs):
        obs = super()._format_raw_obs(raw_obs)
        if self._image_fn is not None:
            if "pixels" in obs:
                obs["pixels"]["top"] = self._corrupt(obs["pixels"]["top"])
            else:
                obs["top"] = self._corrupt(obs["top"])
        return obs

    def render(self):
        return self._corrupt(super().render())
