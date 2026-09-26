"""Summaries of `lerobot-eval` results: success rate with confidence interval
and failure-stage breakdown.

gym-aloha rewards are staged (TransferCube: 1 = touched, 2 = lifted,
3 = transfer attempted, 4 = success), so an episode's max reward tells us
how far the policy got before failing.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

MAX_STAGE = 4


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _find_per_episode(obj: Any) -> list[dict]:
    """Collect all `per_episode` lists, wherever the eval_info layout nests them."""
    found: list[dict] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "per_episode" and isinstance(value, list):
                found.extend(value)
            else:
                found.extend(_find_per_episode(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_find_per_episode(item))
    return found


def merge_eval_infos(paths: list[str | Path]) -> dict:
    """Combine the episodes of several `eval_info.json` files (e.g. eval chunks)."""
    episodes: list[dict] = []
    for path in paths:
        episodes.extend(_find_per_episode(json.loads(Path(path).read_text())))
    return {"per_episode": episodes}


def summarize_eval(eval_info: str | Path | dict) -> dict:
    """Summarize an `eval_info.json` written by `lerobot-eval`."""
    if not isinstance(eval_info, dict):
        eval_info = json.loads(Path(eval_info).read_text())

    episodes = _find_per_episode(eval_info)
    if not episodes:
        raise ValueError("no 'per_episode' entries found in eval_info")

    n = len(episodes)
    successes = sum(bool(ep["success"]) for ep in episodes)
    lo, hi = wilson_ci(successes, n)

    stages = Counter(
        min(MAX_STAGE, max(0, int(round(ep["max_reward"]))))
        for ep in episodes
        if "max_reward" in ep
    )

    return {
        "n_episodes": n,
        "successes": successes,
        "success_rate": successes / n,
        "ci95": (lo, hi),
        "stage_counts": {s: stages.get(s, 0) for s in range(MAX_STAGE + 1)},
    }


def format_summary(summary: dict) -> str:
    lo, hi = summary["ci95"]
    lines = [
        f"Episodes:     {summary['n_episodes']}",
        f"Success rate: {summary['success_rate']:.1%}  (95% CI {lo:.1%} - {hi:.1%})",
        "Max stage reached (0 = nothing, 4 = success):",
    ]
    n = summary["n_episodes"]
    for stage, count in summary["stage_counts"].items():
        lines.append(f"  {stage}: {count:4d}  ({count / n:.1%})")
    return "\n".join(lines)
