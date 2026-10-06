"""Where every policy's checkpoint and results live on Google Drive.

One entry per policy we compare; encoder variants trained with notebook 02
(`VARIANT != "baseline"`) are discovered from their run folders.
"""

from __future__ import annotations

import glob
import os
from typing import NamedTuple

BASE_RUN_NAME = "act_transfer_cube_human_resnet18_s1000"
VARIANT_PREFIX, VARIANT_SUFFIX = "actenc_transfer_cube_human_", "_s1000"
FINAL_STEP = "100000"
# fixed order (and thus fixed figure colors) for the planned variants; others are appended
VARIANT_ORDER = ["dinov2_vits14_frozen", "dinov2_vits14_ft", "resnet18_scratch_ft",
                 "clip_vitb16_frozen", "siglip_vitb16_frozen"]


class PolicyPaths(NamedTuple):
    checkpoint: str  # pretrained_model dir
    robustness: str  # perturbation-suite results (notebook 03)
    clean_eval: str  # 500-episode lerobot-eval (clean success rate)
    probe: str  # feature probe (notebook 04)


def policy_registry(drive_root: str) -> dict[str, PolicyPaths]:
    rob, runs, probe = f"{drive_root}/robustness", f"{drive_root}/runs", f"{drive_root}/probe"
    base = f"{runs}/{BASE_RUN_NAME}"
    registry = {
        "pretrained ACT (LeRobot Hub)": PolicyPaths(
            f"{drive_root}/checkpoints/act_aloha_sim_transfer_cube_human_migrated",
            f"{rob}/pretrained_resnet18",
            f"{drive_root}/eval/pretrained_transfer_cube/n500_seed1000",
            f"{probe}/pretrained_resnet18",
        ),
        "ours, ResNet18 ImageNet (100k)": PolicyPaths(
            f"{base}/checkpoints/{FINAL_STEP}/pretrained_model",
            f"{rob}/ours_resnet18_s1000_100k",
            f"{base}/eval/step{FINAL_STEP}_n500_seed1000",
            f"{probe}/ours_resnet18_s1000_100k",
        ),
    }
    found = {os.path.basename(run).removeprefix(VARIANT_PREFIX).removesuffix(VARIANT_SUFFIX): run
             for run in glob.glob(f"{runs}/{VARIANT_PREFIX}*{VARIANT_SUFFIX}")}
    order = [v for v in VARIANT_ORDER if v in found] + sorted(v for v in found if v not in VARIANT_ORDER)
    for variant in order:
        run = found[variant]
        registry[f"ours, {variant} (100k)"] = PolicyPaths(
            f"{run}/checkpoints/{FINAL_STEP}/pretrained_model",
            f"{rob}/ours_{variant}_s1000_100k",
            f"{run}/eval/step{FINAL_STEP}_n500_seed1000",
            f"{probe}/ours_{variant}_s1000_100k",
        )
    return registry


def trained(registry: dict[str, PolicyPaths]) -> dict[str, PolicyPaths]:
    """Entries whose final checkpoint exists."""
    return {k: v for k, v in registry.items() if os.path.exists(f"{v.checkpoint}/config.json")}


def policy_colors(registry: dict) -> dict[str, str]:
    """A fixed categorical color per policy (registry order), used by every figure so a
    policy keeps its color no matter which subset a plot shows."""
    from avr.analysis.robustness import SERIES_COLORS

    return {label: SERIES_COLORS[i % len(SERIES_COLORS)] for i, label in enumerate(registry)}
