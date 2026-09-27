"""Perturbation suite invariants. Needs gym-aloha (skipped otherwise):

    pip install "gym-aloha==0.1.4" pillow
"""

import numpy as np
import pytest

gym_aloha = pytest.importorskip("gym_aloha")
gym = pytest.importorskip("gymnasium")

from avr.perturb import FACTORS  # noqa: E402
from avr.perturb.env import PerturbedAlohaEnv  # noqa: E402

N_STEPS = 25


@pytest.fixture(scope="module")
def env():
    e = PerturbedAlohaEnv()
    yield e
    e.close()


@pytest.fixture(scope="module")
def actions():
    rng = np.random.default_rng(0)
    return [np.clip(rng.normal(0, 0.3, 14), -1, 1).astype(np.float32) for _ in range(N_STEPS)]


def _rollout(env, seed, actions):
    obs, _ = env.reset(seed=seed)
    frames, states, rewards = [obs["pixels"]["top"]], [obs["agent_pos"]], []
    for a in actions:
        obs, r, *_ = env.step(a)
        frames.append(obs["pixels"]["top"])
        states.append(obs["agent_pos"])
        rewards.append(r)
    return np.stack(frames), np.stack(states), rewards


def test_unperturbed_matches_original_gym_aloha(env, actions):
    orig = gym.make("gym_aloha/AlohaTransferCube-v0", obs_type="pixels_agent_pos")
    env.set_perturbation(None)
    ours = _rollout(env, 5, actions)
    theirs = _rollout(orig.unwrapped, 5, actions)
    assert np.array_equal(ours[0], theirs[0]), "frames differ"
    assert np.array_equal(ours[1], theirs[1]), "states differ"
    assert ours[2] == theirs[2]


@pytest.mark.parametrize("factor", sorted(FACTORS))
def test_factor_changes_pixels_but_not_physics(env, actions, factor):
    env.set_perturbation(None)
    clean_frames, clean_states, clean_rewards = _rollout(env, 5, actions)
    env.set_perturbation(factor, 4)
    frames, states, rewards = _rollout(env, 5, actions)
    assert not np.array_equal(frames[0], clean_frames[0]), "level 4 should change the image"
    assert np.array_equal(states, clean_states), "physics must be unaffected"
    assert rewards == clean_rewards


@pytest.mark.parametrize("factor", ["camera_pose", "distractors", "light_color", "background"])
def test_reset_restores_the_clean_scene(env, factor):
    env.set_perturbation(None)
    clean, _ = env.reset(seed=9)
    env.set_perturbation(factor, 4)
    env.reset(seed=9)
    env.set_perturbation(None)
    again, _ = env.reset(seed=9)
    assert np.array_equal(clean["pixels"]["top"], again["pixels"]["top"])


def test_same_seed_same_perturbation(env):
    env.set_perturbation("distractors", 3)
    a, info_a = env.reset(seed=21)
    b, info_b = env.reset(seed=21)
    c, info_c = env.reset(seed=22)
    assert info_a["perturbation"] == info_b["perturbation"]
    assert np.array_equal(a["pixels"]["top"], b["pixels"]["top"])
    assert info_a["perturbation"] != info_c["perturbation"]


def test_distractors_keep_clear_of_the_cube(env):
    from gym_aloha.utils import sample_box_pose

    env.set_perturbation("distractors", 4)
    for seed in range(20):
        _, info = env.reset(seed=seed)
        cube = sample_box_pose(seed)[:2]
        for obj in info["perturbation"]["params"]["objects"]:
            assert np.linalg.norm(np.array(obj["xy"]) - cube) > 0.15


def test_rendering_off_keeps_physics_and_render_observation_matches(env, actions):
    env.set_perturbation("light_color", 3)
    _, clean_states, _ = _rollout(env, 7, actions)
    env.set_rendering(False)
    try:
        frames, states, _ = _rollout(env, 7, actions)
        assert not frames[1:].any(), "no rendering while off"
        assert np.array_equal(states, clean_states)
        on_demand = env.render_observation()
    finally:
        env.set_rendering(True)
    env.reset(seed=7)
    for a in actions:
        obs, *_ = env.step(a)
    assert np.array_equal(on_demand, obs["pixels"]["top"])
