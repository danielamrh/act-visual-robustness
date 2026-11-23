"""How well does the image-space shadow edit (`avr.shadow_aug`) match real shadows?

In simulation the true shadow mask is known: render the same state with and without shadow-casting lights and take
the difference, restricted to table pixels (segmentation render). States are the arm poses of the real training
demonstrations, so the shadows look like the ones the augmentation will see. Reported per frame:

  IoU / precision / recall of the detected shadow mask vs. the true one (table pixels only)
  removal: fraction of the clean-vs-no-shadow pixel difference on the table that `remove_shadows` takes away
  the same under brightness / contrast jitter applied first (as in training, where ShadowEdit runs after LeRobot's
  color transforms)

    python -m avr.analysis.shadow_validation --frames 40 --out results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from avr.shadow_aug import detect_shadows, remove_shadows, shift_shadows

DATASET = "lerobot/aloha_sim_transfer_cube_human"


def demo_frames(n: int, seed: int = 0):
    """(state, image) pairs spread over the training demonstrations."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(DATASET, video_backend="pyav")
    rng = np.random.default_rng(seed)
    for idx in sorted(rng.choice(len(ds), size=n, replace=False).tolist()):
        item = ds[int(idx)]
        yield item["observation.state"].numpy(), item["observation.images.top"]


def _set_arm_state(physics, state: np.ndarray):
    from gym_aloha.constants import unnormalize_puppet_gripper_position as ungrip

    qpos = physics.data.qpos
    lg, rg = ungrip(state[6]), ungrip(state[13])
    qpos[:16] = np.concatenate([state[:6], [lg, -lg], state[7:13], [rg, -rg]])
    physics.forward()


def render_pair(env, state, cube_seed: int):
    """Clean and shadow-free renders of the same state, plus the table-pixel mask."""
    physics = env._env.physics
    env.set_perturbation(None)
    env.reset(seed=cube_seed)
    _set_arm_state(physics, state)
    kw = dict(height=480, width=640, camera_id="top")
    clean = physics.render(**kw).copy()
    seg = physics.render(segmentation=True, **kw)
    table = (seg[..., 0] == physics.model.name2id("table", "geom")) & (seg[..., 1] == 5)  # mjOBJ_GEOM
    cast = physics.model.light_castshadow.copy()
    physics.model.light_castshadow[...] = 0
    no_shadow = physics.render(**kw).copy()
    physics.model.light_castshadow[...] = cast
    return clean, no_shadow, table


def to_tensor(img_u8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img_u8).permute(2, 0, 1).float() / 255


def jitter(img: torch.Tensor, brightness: float, contrast: float) -> torch.Tensor:
    from torchvision.transforms.v2 import functional as TF

    return TF.adjust_contrast(TF.adjust_brightness(img, brightness), contrast)


def frame_metrics(clean: torch.Tensor, no_shadow: torch.Tensor, table: torch.Tensor) -> dict:
    gt = ((clean - no_shadow).abs().amax(0) > 8 / 255) & table
    det = detect_shadows(clean)
    pred = det["shadow"] & table
    inter, union = (pred & gt).sum().item(), (pred | gt).sum().item()
    before = (clean - no_shadow).abs()[:, table].mean().item()
    after = (remove_shadows(clean, det) - no_shadow).abs()[:, table].mean().item()
    return {
        "iou": inter / union if union else 1.0,
        "precision": inter / max(pred.sum().item(), 1),
        "recall": inter / max(gt.sum().item(), 1),
        "gt_pixels": int(gt.sum()),
        "removed": 1 - after / before if before > 0 else 1.0,
    }


def validate(n_frames: int = 40, seed: int = 0):
    from avr.perturb.env import PerturbedAlohaEnv

    env = PerturbedAlohaEnv()
    rows, examples = [], []
    rng = np.random.default_rng(seed)
    for i, (state, demo_img) in enumerate(demo_frames(n_frames, seed)):
        clean_u8, no_shadow_u8, table_np = render_pair(env, state, cube_seed=1000 + i)
        clean, no_shadow, table = to_tensor(clean_u8), to_tensor(no_shadow_u8), torch.from_numpy(table_np)
        row = {"frame": i, **frame_metrics(clean, no_shadow, table)}
        b, c = rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)  # LeRobot's brightness / contrast ranges
        jm = frame_metrics(jitter(clean, b, c), jitter(no_shadow, b, c), table)
        row.update({f"jitter_{k}": v for k, v in jm.items() if k != "gt_pixels"})
        rows.append(row)
        if i < 3:
            examples.append((clean, no_shadow, demo_img))
    env.close()
    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ("frame", "gt_pixels")}
    return rows, summary, examples


def example_figure(examples, path):
    import matplotlib.pyplot as plt

    from avr.analysis.robustness import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY

    cols = ["sim, clean", "sim, true shadows off", "sim, our removal", "demo frame", "demo, removal", "demo, shifted"]
    fig, axes = plt.subplots(len(examples), len(cols), figsize=(2.6 * len(cols), 2.1 * len(examples)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for r, (clean, no_shadow, demo) in enumerate(examples):
        d_sim, d_demo = detect_shadows(clean), detect_shadows(demo)
        h, w = demo.shape[-2:]
        imgs = [clean, no_shadow, remove_shadows(clean, d_sim), demo, remove_shadows(demo, d_demo),
                shift_shadows(demo, d_demo, round(0.06 * h), round(-0.06 * w))]  # fmt: skip
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], fontsize=9, color=TEXT_PRIMARY)
    axes[0, 0].set_ylabel("example 1", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle("Shadow edit: simulation ground truth (left) and recorded demonstration frames (right)",
                 x=0.01, ha="left", fontsize=11, color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=130, facecolor=SURFACE)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    rows, summary, examples = validate(args.frames)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "shadow_validation.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=1))
    example_figure(examples, out / "shadow_edit_examples.png")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
