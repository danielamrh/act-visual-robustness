"""Evaluate a policy in chunks of episodes, one `lerobot-eval` process each.

Each chunk uses its own seed range (seed + chunk start), so N chunks of size
C cover the same episodes as one run of N*C episodes. Finished chunks are
kept on Drive and skipped when the cell is re-run, so a crash or a lost
Colab session costs at most one chunk.
"""

from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

from avr.background import start_background, watch
from avr.lerobot_cli import eval_cmd


def chunk_info_paths(eval_root: str | Path) -> list[str]:
    return sorted(glob.glob(f"{eval_root}/chunk*/**/eval_info.json", recursive=True))


def run_chunked_eval(
    policy_path: str,
    eval_root: str,
    n_episodes: int,
    chunk_size: int = 100,
    seed: int = 1000,
    task: str = "AlohaTransferCube-v0",
    batch_size: int = 5,
    log_dir: str = "/content/logs",
    interval: float = 60.0,
) -> list[str]:
    """Run missing chunks and return the `eval_info.json` paths of all chunks."""
    for k, start in enumerate(range(0, n_episodes, chunk_size)):
        n = min(chunk_size, n_episodes - start)
        out_dir = f"{eval_root}/chunk{k:02d}"
        if glob.glob(f"{out_dir}/**/eval_info.json", recursive=True):
            print(f"chunk {k}: done, skipping")
            continue
        if os.path.exists(out_dir):  # left over from a crashed run
            shutil.rmtree(out_dir)
        print(f"chunk {k}: episodes {start}-{start + n - 1} (seed {seed + start})", flush=True)
        cmd = eval_cmd(policy_path, out_dir, task=task, n_episodes=n, batch_size=batch_size, seed=seed + start)
        log_path = f"{log_dir}/{Path(eval_root).name}_chunk{k:02d}.log"
        watch(start_background(cmd, log_path), log_path, interval=interval)
    return chunk_info_paths(eval_root)
