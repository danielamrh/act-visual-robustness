import torch

from avr.shadow_aug import ShadowEdit, detect_shadows, lerobot_flags, remove_shadows, shift_shadows

TABLE, FACTOR = 0.30, 0.83


def scene():
    """Black background, grey table, a square shadow on it and a red cube."""
    img = torch.zeros(3, 120, 160)
    img[:, 20:100, 20:140] = TABLE
    img[:, 50:70, 60:90] = TABLE * FACTOR
    img[:, 40:46, 100:106] = torch.tensor([0.8, 0.1, 0.1])[:, None, None]
    return img


def test_detects_and_removes_the_shadow():
    img = scene()
    det = detect_shadows(img)
    truth = torch.zeros(120, 160, dtype=torch.bool)
    truth[50:70, 60:90] = True
    assert (det["shadow"] == truth).all()
    out = remove_shadows(img, det)
    table = out[:, 50:100, 20:140]  # below the cube
    assert torch.allclose(table, torch.full_like(table, TABLE), atol=0.01)
    assert torch.equal(out[:, 40:46, 100:106], img[:, 40:46, 100:106])  # the cube is untouched


def test_detection_survives_brightness_and_contrast_jitter():
    from torchvision.transforms.v2 import functional as TF

    img = TF.adjust_contrast(TF.adjust_brightness(scene(), 1.2), 0.8)
    assert detect_shadows(img)["shadow"].sum() == 20 * 30


def test_shift_moves_the_shadow_on_the_table_only():
    img = scene()
    out = shift_shadows(img, detect_shadows(img), dy=0, dx=-30)
    assert out[:, 60, 45].max() < 0.95 * TABLE  # new shadow position
    assert abs(out[:, 60, 85].max() - TABLE) < 0.01  # old position is lit again
    assert torch.equal(out[:, :20], img[:, :20])  # background unchanged


def test_transform_handles_batches_and_skips_frames():
    torch.manual_seed(0)
    imgs = torch.stack([scene(), scene()])
    assert torch.equal(ShadowEdit(p=0.0)(imgs), imgs)
    out = ShadowEdit(p=1.0, p_remove=1.0, strength=(1.0, 1.0))(imgs)
    assert out.shape == imgs.shape and not torch.equal(out, imgs)


def test_lerobot_flags_build_run_a_plus_shadow_edit():
    import draccus
    from dataclasses import dataclass, field

    from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig

    from avr.shadow_aug import register

    @dataclass
    class Cfg:
        image_transforms: ImageTransformsConfig = field(default_factory=ImageTransformsConfig)

    register()
    args = [f.replace("--dataset.", "--") for f in lerobot_flags()]
    cfg = draccus.parse(Cfg, args=args).image_transforms
    assert list(cfg.tfs) == [*ImageTransformsConfig().tfs, "shadow"]  # LeRobot's defaults kept, shadow last
    tf = ImageTransforms(cfg)
    assert isinstance(tf.transforms["shadow"], ShadowEdit)
    # the shadow edit is always among the sampled transforms, and three others as in run A
    for _ in range(200):
        tf(scene())
        assert len(tf.tf.selected_transforms) == 4 and isinstance(tf.tf.selected_transforms[-1], ShadowEdit)
