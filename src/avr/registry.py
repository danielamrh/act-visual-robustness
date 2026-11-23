"""Where every policy's checkpoint and results live on Google Drive.

One entry per policy we compare; encoder variants trained with notebook 02
(`VARIANT != "baseline"`) are discovered from their run folders.
"""

from __future__ import annotations

import glob
import os
from typing import NamedTuple

BASE_RUN_NAME = "act_transfer_cube_human_resnet18_s1000"
AUG_RUN_NAME = "act_transfer_cube_human_resnet18_aug_s1000"  # baseline recipe + LeRobot image augmentation
SHADOW_AUG_RUN_NAME = "act_transfer_cube_human_resnet18_shadowaug_s1000"  # run A + shadow removal / shift
VARIANT_PREFIX, VARIANT_SUFFIX = "actenc_transfer_cube_human_", "_s1000"
FINAL_STEP = "100000"
# fixed order (and thus fixed figure colors) for the planned variants; others are appended
VARIANT_ORDER = ["dinov2_vits14_frozen", "dinov2_vits14_ft", "resnet18_scratch_ft",
                 "clip_vitb16_frozen", "siglip_vitb16_frozen", "dinov2_vits14_hr_frozen"]


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
    aug = f"{runs}/{AUG_RUN_NAME}"
    registry["ours, ResNet18 ImageNet + augmentation (100k)"] = PolicyPaths(
        f"{aug}/checkpoints/{FINAL_STEP}/pretrained_model",
        f"{rob}/ours_resnet18_aug_s1000_100k",
        f"{aug}/eval/step{FINAL_STEP}_n500_seed1000",
        f"{probe}/ours_resnet18_aug_s1000_100k",
    )
    shadow = f"{runs}/{SHADOW_AUG_RUN_NAME}"
    registry["ours, ResNet18 ImageNet + augmentation + shadow edit (100k)"] = PolicyPaths(
        f"{shadow}/checkpoints/{FINAL_STEP}/pretrained_model",
        f"{rob}/ours_resnet18_shadowaug_s1000_100k",
        f"{shadow}/eval/step{FINAL_STEP}_n500_seed1000",
        f"{probe}/ours_resnet18_shadowaug_s1000_100k",
    )
    return registry


def trained(registry: dict[str, PolicyPaths]) -> dict[str, PolicyPaths]:
    """Entries whose final checkpoint exists."""
    return {k: v for k, v in registry.items() if os.path.exists(f"{v.checkpoint}/config.json")}


# Fixed colors (validated categorical palette, light surface) for the policies we report, chosen so that the
# pairs compared most (baseline vs. its augmented version) are far apart; anything else gets the remaining slots.
FIXED_COLORS = {
    "pretrained ACT (LeRobot Hub)": "#2a78d6",                   # blue
    "ours, ResNet18 ImageNet (100k)": "#eb6834",                 # orange
    "ours, dinov2_vits14_frozen (100k)": "#1baf7a",              # aqua
    "ours, dinov2_vits14_ft (100k)": "#eda100",                  # yellow
    "ours, ResNet18 ImageNet + augmentation (100k)": "#4a3aa7",  # violet
    "ours, dinov2_vits14_hr_frozen (100k)": "#e87ba4",           # magenta
    "ours, ResNet18 ImageNet + augmentation + shadow edit (100k)": "#008300",  # green
}


def policy_colors(registry: dict) -> dict[str, str]:
    """A fixed categorical color per policy, used by every figure so a policy keeps its color no
    matter which subset a plot shows (or which empty run folders exist on Drive)."""
    from avr.analysis.robustness import SERIES_COLORS

    spare = [c for c in SERIES_COLORS if c not in FIXED_COLORS.values()]
    others = [label for label in registry if label not in FIXED_COLORS]
    colors = {label: FIXED_COLORS[label] for label in registry if label in FIXED_COLORS}
    colors.update({label: spare[i % len(spare)] for i, label in enumerate(others)})
    return colors
