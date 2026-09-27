import numpy as np
import pytest

from avr.analysis.attention import cell_means, load_figure_maps, shift_table
from avr.registry import policy_registry, trained


def test_registry_discovers_variants(tmp_path):
    root = tmp_path / "drive"
    variant_run = root / "runs" / "actenc_transfer_cube_human_dinov2_vits14_frozen_s1000"
    (variant_run / "checkpoints" / "100000" / "pretrained_model").mkdir(parents=True)
    (variant_run / "checkpoints" / "100000" / "pretrained_model" / "config.json").write_text("{}")
    reg = policy_registry(str(root))
    label = "ours, dinov2_vits14_frozen (100k)"
    assert label in reg
    assert reg[label].robustness.endswith("robustness/ours_dinov2_vits14_frozen_s1000_100k")
    assert reg[label].clean_eval.endswith("eval/step100000_n500_seed1000")
    assert list(trained(reg)) == [label]  # the others have no checkpoint here


def _rows():
    rows = []
    for seed, base in ((1, 0.1), (2, 0.3)):
        for step in (0, 50):
            rows.append({"seed": seed, "factor": "noise", "level": 2, "step": step,
                         "feature_shift": base + step / 1000, "action_shift": 1.0, "attn_shift": 0.0})
    return rows


def test_cell_means_average_steps_then_seeds():
    mean, se = cell_means(_rows(), "feature_shift")[("noise", 2)]
    per_seed = np.array([0.125, 0.325])  # each seed: mean over its two probe steps
    assert mean == pytest.approx(per_seed.mean())
    assert se == pytest.approx(per_seed.std(ddof=1) / np.sqrt(2))
    assert shift_table(_rows())[0]["action_shift"] == 1.0


def test_load_figure_maps_roundtrip(tmp_path):
    path = tmp_path / "figures_seed1000.npz"
    np.savez_compressed(path, **{
        "clean__step50__img": np.zeros((4, 4, 3), np.uint8),
        "clean__step50__occl": np.ones((2, 2)),
        "noise_L3__step50__attn": np.ones((2, 2)),
    })
    maps = load_figure_maps(path)
    assert set(maps) == {"None_L0", "noise_L3"}
    assert set(maps["None_L0"][50]) == {"img", "occl"}
