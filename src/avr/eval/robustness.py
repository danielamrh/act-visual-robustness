"""Evaluate a LeRobot policy under the visual perturbation suite.

Mirrors `lerobot-eval`'s rollout (same processors, same per-step order), but
steps our `PerturbedAlohaEnv` one episode at a time. Each (factor, level) cell
is written to `<out>/<factor>/level<k>.json`; existing cells are skipped, so
an interrupted run resumes where it stopped.

    python -m avr.eval.robustness --policy <dir> --out <dir> \\
        --factors light_color noise --levels 1 2 3 4 --episodes 50
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from avr.perturb.suite import FACTORS

MAX_STEPS = 400  # lerobot's aloha episode_length
TASKS = {"AlohaTransferCube-v0": "transfer_cube", "AlohaInsertion-v0": "insertion"}


@dataclass
class LoadedPolicy:
    policy: Any
    preprocessor: Any
    postprocessor: Any
    env_preprocessor: Any
    env_postprocessor: Any


def load_policy(policy_path: str, task: str = "AlohaTransferCube-v0", device: str = "cuda") -> LoadedPolicy:
    """Load policy + processors exactly like lerobot-eval (lerobot 0.6.1)."""
    from lerobot.configs import PreTrainedConfig
    from lerobot.envs import make_env_pre_post_processors
    from lerobot.envs.configs import AlohaEnv as AlohaEnvConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.pretrained_path = policy_path
    cfg.device = device
    env_cfg = AlohaEnvConfig(task=task)
    policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map={})
    policy.eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": {}},
        },
    )
    env_pre, env_post = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg)
    return LoadedPolicy(policy, pre, post, env_pre, env_post)


def run_episode(lp: LoadedPolicy, env, seed: int, record_every: int = 0) -> tuple[dict, list]:
    """One episode. Returns (record, frames); frames only if `record_every` > 0."""
    import torch
    from lerobot.envs import preprocess_observation
    from lerobot.utils.constants import ACTION

    obs, info = env.reset(seed=seed)
    lp.policy.reset()
    rewards, frames = [], []
    success = False
    for step in range(MAX_STEPS):
        if record_every and step % record_every == 0:
            frames.append(obs["pixels"]["top"])
        batch = {"pixels": {"top": obs["pixels"]["top"][None]}, "agent_pos": obs["agent_pos"][None]}
        observation = preprocess_observation(batch)
        observation["task"] = [""]
        observation = lp.env_preprocessor(observation)
        observation = lp.preprocessor(observation)
        with torch.inference_mode():
            action = lp.policy.select_action(observation)
        action = lp.postprocessor(action)
        action = lp.env_postprocessor({ACTION: action})[ACTION]
        obs, reward, terminated, truncated, _ = env.step(action.to("cpu").numpy()[0])
        rewards.append(float(reward))
        if terminated:
            success = True
            break
    record = {
        "seed": seed,
        "success": success,
        "max_reward": max(rewards),
        "sum_reward": sum(rewards),
        "steps": len(rewards),
        "perturbation": info["perturbation"],
    }
    return record, frames


def _write_video(path: Path, frames: list, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [f[::2, ::2] for f in frames], fps=fps, macro_block_size=8)


def evaluate_cell(lp, env, factor, level, seeds, videos: int, video_dir: Path | None) -> dict:
    env.set_perturbation(factor, level)
    records = []
    t0 = time.time()
    for i, seed in enumerate(seeds):
        want_video = video_dir is not None and i < videos
        record, frames = run_episode(lp, env, seed, record_every=2 if want_video else 0)
        records.append(record)
        if want_video:
            tag = "ok" if record["success"] else "fail"
            _write_video(video_dir / f"{factor}_level{level}_seed{seed}_{tag}.mp4", frames, fps=25)
        n_ok = sum(r["success"] for r in records)
        print(f"{factor or 'clean'} L{level}  episode {i + 1}/{len(seeds)}  success so far {n_ok}/{i + 1}", flush=True)
    return {
        "factor": factor,
        "level": level,
        "category": FACTORS[factor].category if factor else "clean",
        "n_episodes": len(records),
        "success_rate": sum(r["success"] for r in records) / len(records),
        "eval_s": time.time() - t0,
        "episodes": records,
    }


def cell_path(out: Path, factor: str | None, level: int) -> Path:
    return out / (factor or "clean") / f"level{level}.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="AlohaTransferCube-v0")
    ap.add_argument("--factors", nargs="+", default=sorted(FACTORS), help="factor names, or 'clean'")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1000, help="episode i uses seed + i (as lerobot-eval)")
    ap.add_argument("--videos", type=int, default=2, help="videos saved per cell")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from avr.perturb.env import PerturbedAlohaEnv

    out = Path(args.out)
    seeds = list(range(args.seed, args.seed + args.episodes))
    cells = [(None, 0)] if args.factors == ["clean"] else [(f, lv) for f in args.factors for lv in args.levels]
    todo = [(f, lv) for f, lv in cells if not cell_path(out, f, lv).exists()]
    print(f"{len(cells) - len(todo)} of {len(cells)} cells already done; running {len(todo)}", flush=True)
    if not todo:
        return

    lp = load_policy(args.policy, task=args.task, device=args.device)
    env = PerturbedAlohaEnv(task=TASKS[args.task])
    for factor, level in todo:
        result = evaluate_cell(lp, env, factor, level, seeds, args.videos, out / "videos")
        result.update(policy=args.policy, seed=args.seed)
        path = cell_path(out, factor, level)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial")
        tmp.write_text(json.dumps(result, indent=1))
        tmp.rename(path)
        print(f"== {factor or 'clean'} level {level}: {result['success_rate']:.0%} "
              f"({result['n_episodes']} episodes, {result['eval_s'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
