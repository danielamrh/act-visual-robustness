"""Training data for run C: run A's data (the 50 recorded demonstrations) plus shadow-free copies of them.

The 50 demonstrations of `lerobot/aloha_sim_transfer_cube_human` were recorded in the same simulation we evaluate in,
so they can be re-created: the cube's start position is read from the first frame (the cube is the only red object;
a homography from renders at known positions maps its pixel center to the table), then the recorded actions are
replayed and rendered with the shadow-casting light switched off (exactly the suite's `shadows_off`). States and
actions are the recorded ones.

Every frame of the replay is also rendered *with* shadows and compared with the recording. Contacts make the replay
drift for a few demonstrations (7 of 50 in our check, cube up to 108 px off); a demonstration whose cube leaves the
recording by more than `MAX_CUBE_DIFF_PX` in any frame gets no shadow-free copy, so images never contradict the
recorded states and actions. The recorded demonstrations are all kept, so run C sees run A's data plus the copies.

    python -m avr.noshadow_data --root /tmp/avr/data/shadowpairs [--episodes 0 1]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SOURCE = "lerobot/aloha_sim_transfer_cube_human"
REPO_ID = "local/aloha_sim_transfer_cube_human_shadowpairs"
MAX_CUBE_DIFF_PX = 8.0  # of 640 (the cube is ~15 px wide); 43 of 50 replays stay within this
RENDER = dict(height=480, width=640, camera_id="top")


def red_center(img: np.ndarray) -> tuple[np.ndarray, int]:
    """Pixel center (x, y) and size of the red cube in an (H, W, 3) uint8 image."""
    img = img.astype(np.int16)
    m = (img[..., 0] > 100) & (img[..., 0] - img[..., 1] > 50) & (img[..., 0] - img[..., 2] > 50)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return np.array([np.nan, np.nan]), 0
    return np.array([xs.mean(), ys.mean()]), int(m.sum())


def _homography(world, pix) -> np.ndarray:
    rows = []
    for (x, y), (u, v) in zip(world, pix):
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    h = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    return h / h[2, 2]


def _apply(h: np.ndarray, p) -> np.ndarray:
    q = h @ np.array([p[0], p[1], 1.0])
    return q[:2] / q[2]


class Replayer:
    """Re-creates a demonstration in gym-aloha and renders it with and without shadows."""

    def __init__(self):
        from avr.perturb.env import PerturbedAlohaEnv

        self.env = PerturbedAlohaEnv()  # the evaluation env, unperturbed
        self.env.set_rendering(False)  # the env's own per-step render is not needed: we render ourselves
        self.dm = self.env._env
        self.physics = self.dm.physics
        grid = [(x, y) for x in (0.0, 0.1, 0.2) for y in (0.4, 0.5, 0.6)]  # gym-aloha's cube start range
        pix = []
        for x, y in grid:
            self._reset(x, y)
            pix.append(red_center(self.render(shadows=True))[0])
        self.h_inv = np.linalg.inv(_homography(grid, pix))

    def _reset(self, x: float, y: float):
        from gym_aloha.tasks.sim import BOX_POSE

        BOX_POSE[0] = np.array([x, y, 0.05, 1, 0, 0, 0])  # gym-aloha's cubes start upright at z = 0.05
        self.dm.reset()

    def render(self, shadows: bool) -> np.ndarray:
        cast = self.physics.model.light_castshadow.copy()
        if not shadows:
            self.physics.model.light_castshadow[...] = 0
        img = self.physics.render(**RENDER).copy()
        self.physics.model.light_castshadow[...] = cast
        return img

    def start_position(self, frame0: np.ndarray, iters: int = 3) -> np.ndarray:
        """Cube start (x, y) from the first recorded frame; refined by rendering at the estimate."""
        c0 = red_center(frame0)[0]
        target = _apply(self.h_inv, c0)
        xy = target.copy()
        for _ in range(iters):
            self._reset(*xy)
            xy = xy + target - _apply(self.h_inv, red_center(self.render(shadows=True))[0])
        return xy

    def replay(self, xy, actions):
        """Yield (with shadows, without shadows, reward) for frame 0 and after each action."""
        self._reset(*xy)
        reward = 0
        yield self.render(True), self.render(False), reward
        for a in actions:
            ts = self.dm.step(a)
            reward = max(reward, ts.reward or 0)
            yield self.render(True), self.render(False), reward

    def close(self):
        self.env.close()


def _frame(ds, idx: int) -> np.ndarray:
    return (ds[idx]["observation.images.top"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


def build(root: str, episodes: list[int] | None = None, log=print) -> dict:
    """Write the dataset (recorded demonstrations + shadow-free copies) to `root`; returns a fidelity report."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(SOURCE, video_backend="pyav")
    hf = src.hf_dataset.with_format("numpy")
    keep = ("observation.images.top", "observation.state", "action")
    features = {k: v for k, v in src.meta.features.items() if k in keep}
    dst = LeRobotDataset.create(REPO_ID, fps=src.fps, features=features, root=root, use_videos=True,
                                robot_type=src.meta.robot_type)
    rep = Replayer()
    report = {"source": SOURCE, "max_cube_diff_px": MAX_CUBE_DIFF_PX, "copied": [], "not_copied": []}
    episodes = list(range(src.num_episodes)) if episodes is None else episodes
    t_start = time.time()

    def save(images, states, actions, task):
        for t, img in enumerate(images):
            dst.add_frame({"observation.images.top": img, "observation.state": states[t], "action": actions[t],
                           "task": task})
        dst.save_episode()

    for ep in episodes:
        meta = src.meta.episodes[ep]
        lo, hi = meta["dataset_from_index"], meta["dataset_to_index"]
        task = meta["tasks"][0]
        states, actions = hf[lo:hi]["observation.state"], hf[lo:hi]["action"]
        recorded = [_frame(src, i) for i in range(lo, hi)]
        xy = rep.start_position(recorded[0])
        without_s, diffs, maes = [], [], []
        for t, (img_s, img_n, reward) in enumerate(rep.replay(xy, actions[:-1])):
            without_s.append(img_n)
            diffs.append(float(np.linalg.norm(red_center(img_s)[0] - red_center(recorded[t])[0])))
            maes.append(float(np.abs(img_s.astype(np.int16) - recorded[t].astype(np.int16)).mean()))
        row = {"episode": ep, "cube_start": [float(xy[0]), float(xy[1])], "replay_success": bool(reward == 4),
               "max_cube_diff_px": float(np.nanmax(diffs)), "mean_image_mae": float(np.mean(maes))}
        row["copied"] = row["max_cube_diff_px"] <= MAX_CUBE_DIFF_PX  # False also for NaN (cube not found)
        save(recorded, states, actions, task)  # run A's data, unchanged
        if row["copied"]:
            save(without_s, states, actions, task)
        report["copied" if row["copied"] else "not_copied"].append(row)
        log(f"episode {ep}: cube diff max {row['max_cube_diff_px']:.1f} px, image MAE {row['mean_image_mae']:.2f}"
            f"/255, replay success {row['replay_success']}, shadow-free copy: {row['copied']} "
            f"({time.time() - t_start:.0f} s)", flush=True)
    dst.finalize()
    rep.close()
    report["n_recorded"], report["n_shadow_free"] = len(episodes), len(report["copied"])
    (Path(root) / "build_report.json").write_text(json.dumps(report, indent=1))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    args = ap.parse_args(argv)
    report = build(args.root, args.episodes)
    print(f"{report['n_recorded']} recorded + {report['n_shadow_free']} shadow-free demonstrations written to "
          f"{args.root}; no copy (replay drifted): {[r['episode'] for r in report['not_copied']]}")


if __name__ == "__main__":
    main()
