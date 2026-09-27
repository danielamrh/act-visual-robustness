"""Why does a perturbation hurt? Probe what changes inside the policy.

Because every perturbation is purely visual, a clean episode's actions can be
replayed under any perturbation and the robot passes through *exactly* the same
states; only the camera image differs. At a few probe steps we feed the clean
and the perturbed image (same state) through the policy and measure

  - feature shift: cosine distance between the encoder's feature maps
    (mean over spatial tokens) - how much the representation moves
  - action shift: mean absolute difference of the predicted 100-step action
    chunks (radians / gripper units) - how much the decision moves
  - attention: ACT's decoder cross-attention over the image tokens (mean over
    the 100 action queries and all heads), as an h x w map. In practice it is
    diffuse and piles up on empty background tokens ("attention sinks"), so
    for figures we also compute
  - occlusion sensitivity: how much the action chunk changes when a gray patch
    hides each image location (first seed only, ~200 forward passes per frame)

No closed-loop rollouts are needed under perturbation, so a whole policy
takes minutes. The cross-attention weights are read with a forward hook:
ACT calls `nn.MultiheadAttention` with its default `need_weights=True` and
simply discards them.

    python -m avr.analysis.features --policy <dir> --out <dir> --seeds 10
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from avr.perturb.suite import FACTORS

PROBE_STEPS = (0, 50, 100, 150, 200, 250, 300, 350)
FIGURE_CELLS = [  # conditions whose images + attention maps are kept for figures
    (None, 0), ("shadows_off", 1), ("light_direction", 2), ("noise", 3), ("blur", 3), ("cube_color", 4),
]


@contextmanager
def probe_hooks(policy):
    """Capture the encoder feature map and the decoder cross-attention of one forward pass."""
    model = policy.model
    captured = {}

    def on_backbone(_, __, out):
        captured["fmap"] = out["feature_map"].detach()

    def on_cross_attn(_, __, out):
        captured["attn"] = out[1].detach()  # (B, n_queries, n_memory), averaged over heads

    handles = [
        model.backbone.register_forward_hook(on_backbone),
        model.decoder.layers[-1].multihead_attn.register_forward_hook(on_cross_attn),
    ]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def n_prefix_tokens(policy) -> int:
    """Memory tokens before the image tokens: latent, robot state, env state (if used)."""
    cfg = policy.config
    return 1 + int(cfg.robot_state_feature is not None) + int(cfg.env_state_feature is not None)


def probe(lp, image: np.ndarray, state: np.ndarray) -> dict:
    """One forward pass on (image, state): feature map, action chunk, attention over image tokens."""
    import torch
    from lerobot.envs import preprocess_observation

    batch = preprocess_observation({"pixels": {"top": image[None]}, "agent_pos": state[None]})
    batch["task"] = [""]
    batch = lp.preprocessor(lp.env_preprocessor(batch))
    with probe_hooks(lp.policy) as cap, torch.inference_mode():
        actions = lp.policy.predict_action_chunk(batch)
    actions = lp.postprocessor(actions)  # unnormalized joint targets
    fmap = cap["fmap"][0].float().cpu()  # (C, h, w)
    h, w = fmap.shape[1:]
    attn = cap["attn"][0, :, n_prefix_tokens(lp.policy):].float().mean(0).reshape(h, w).cpu().numpy()
    return {"fmap": fmap, "actions": actions[0].float().cpu().numpy(), "attn": attn}


def _action_chunks(lp, images: np.ndarray, state: np.ndarray):
    """Unnormalized action chunks for a batch of images (N, H, W, 3) sharing one robot state."""
    import torch
    from lerobot.envs import preprocess_observation

    n = len(images)
    batch = preprocess_observation({"pixels": {"top": images}, "agent_pos": np.repeat(state[None], n, 0)})
    batch["task"] = [""] * n
    batch = lp.preprocessor(lp.env_preprocessor(batch))
    with torch.inference_mode():
        return lp.postprocessor(lp.policy.predict_action_chunk(batch)).float().cpu().numpy()


def occlusion_map(lp, image: np.ndarray, state: np.ndarray, patch: int = 40, batch_size: int = 32) -> np.ndarray:
    """Occlusion sensitivity: mean |change| of the predicted action chunk when a gray `patch` x `patch`
    square hides each image location. Returns an (H / patch, W / patch) map."""
    h, w = image.shape[0] // patch, image.shape[1] // patch
    ref = _action_chunks(lp, image[None], state)[0]
    fill = image.reshape(-1, 3).mean(0).astype(np.uint8)
    cells = [(i, j) for i in range(h) for j in range(w)]
    out = np.zeros((h, w), dtype=np.float32)
    for k in range(0, len(cells), batch_size):
        chunk = cells[k : k + batch_size]
        imgs = np.repeat(image[None], len(chunk), 0)
        for n, (i, j) in enumerate(chunk):
            imgs[n, i * patch : (i + 1) * patch, j * patch : (j + 1) * patch] = fill
        acts = _action_chunks(lp, imgs, state)
        for n, (i, j) in enumerate(chunk):
            out[i, j] = np.abs(acts[n] - ref).mean()
    return out


def feature_shift(a, b) -> float:
    """Mean cosine distance between corresponding spatial tokens of two (C, h, w) maps."""
    import torch.nn.functional as F

    return float(1 - F.cosine_similarity(a.flatten(1), b.flatten(1), dim=0).mean())


def record_clean(lp, env, seed: int) -> tuple[list[np.ndarray], bool]:
    """Closed-loop clean episode; returns the executed actions and success."""
    from avr.eval.robustness import MAX_STEPS

    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    env.set_perturbation(None)
    obs, _ = env.reset(seed=seed)
    lp.policy.reset()
    actions, success = [], False
    for _ in range(MAX_STEPS):
        batch = preprocess_observation({"pixels": {"top": obs["pixels"]["top"][None]}, "agent_pos": obs["agent_pos"][None]})
        batch["task"] = [""]
        batch = lp.preprocessor(lp.env_preprocessor(batch))
        with torch.inference_mode():
            action = lp.policy.select_action(batch)
        action = lp.env_postprocessor({ACTION: lp.postprocessor(action)})[ACTION].cpu().numpy()[0]
        actions.append(action)
        obs, _, terminated, _, _ = env.step(action)
        if terminated:
            success = True
            break
    return actions, success


def replay_frames(env, factor, level, seed, actions, probe_steps) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Replay `actions` under a perturbation; return {step: (image, state)} at the probe steps."""
    env.set_perturbation(factor, level)
    env.set_rendering(False)
    frames = {}
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max(probe_steps) + 1):
            if step in probe_steps:
                frames[step] = (env.render_observation(), obs["agent_pos"].copy())
            if step >= len(actions):
                break
            obs, *_ = env.step(actions[step])
    finally:
        env.set_rendering(True)
    return frames


def analyze_seed(lp, env, seed, cells, probe_steps=PROBE_STEPS, keep_figures=False):
    actions, success = record_clean(lp, env, seed)
    steps = [s for s in probe_steps if s < len(actions)]
    clean = {s: probe(lp, *img_state) for s, img_state in replay_frames(env, None, 0, seed, actions, steps).items()}
    rows, figures = [], {}
    fig_steps = steps[1:3]  # e.g. steps 50 and 100: arms moving towards the cube
    if keep_figures:
        clean_frames = replay_frames(env, None, 0, seed, actions, fig_steps)
        figures["clean"] = {s: _figure_entry(lp, img, st, clean[s]["attn"]) for s, (img, st) in clean_frames.items()}
        # shadow mask: pixels that change when shadows are switched off in the same state
        for s, (img_off, _) in replay_frames(env, "shadows_off", 1, seed, actions, fig_steps).items():
            diff = np.abs(clean_frames[s][0].astype(int) - img_off.astype(int)).max(-1)
            figures["clean"][s]["shadow_mask"] = (diff > 8).astype(np.uint8)
    for factor, level in cells:
        frames = replay_frames(env, factor, level, seed, actions, steps)
        for step, (img, state) in frames.items():
            p = probe(lp, img, state)
            rows.append({
                "seed": seed, "clean_success": success, "factor": factor, "level": level, "step": step,
                "feature_shift": feature_shift(clean[step]["fmap"], p["fmap"]),
                "action_shift": float(np.abs(p["actions"] - clean[step]["actions"]).mean()),
                "attn_shift": float(0.5 * np.abs(p["attn"] - clean[step]["attn"]).sum()),  # total variation
            })  # fmt: skip
            if keep_figures and (factor, level) in FIGURE_CELLS and step in fig_steps:
                figures.setdefault(f"{factor}_L{level}", {})[step] = _figure_entry(lp, img, state, p["attn"])
    return rows, figures


def _figure_entry(lp, img, state, attn) -> dict:
    return {"img": img, "attn": attn, "occl": occlusion_map(lp, img, state)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds, starting at --seed")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--factors", nargs="+", default=sorted(FACTORS))
    args = ap.parse_args(argv)

    from avr.eval.robustness import load_policy
    from avr.perturb.env import PerturbedAlohaEnv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cells = [(f, lv) for f in args.factors for lv in ([1] if f == "shadows_off" else [1, 2, 3, 4])]
    lp = load_policy(args.policy, device=args.device)
    env = PerturbedAlohaEnv()
    rows_path = out / "feature_probe.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(line)["seed"] for line in rows_path.read_text().splitlines() if line}
    for i, seed in enumerate(range(args.seed, args.seed + args.seeds)):
        if seed in done:
            print(f"seed {seed}: done, skipping", flush=True)
            continue
        t0 = time.time()
        rows, figures = analyze_seed(lp, env, seed, cells, keep_figures=(i == 0))
        with open(rows_path, "a") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        if figures:
            np.savez_compressed(
                out / f"figures_seed{seed}.npz",
                **{f"{name}__step{s}__{kind}": arr for name, per_step in figures.items()
                   for s, entry in per_step.items() for kind, arr in entry.items()},
            )  # fmt: skip
        print(f"seed {seed}: {len(rows)} probes in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
