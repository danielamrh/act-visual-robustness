"""Encoder wrappers. Needs torch, torchvision and timm (skipped otherwise).
Weights are not downloaded here (pretrained=False); shapes do not depend on them."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("torchvision")

from avr.encoders import IMAGENET_MEAN, IMAGENET_STD, build_encoder  # noqa: E402

EXPECTED = {
    "resnet18_scratch": (512, 15, 20),
    "dinov2_vits14": (384, 16, 21),
    "dinov2_vits14_hr": (384, 16, 21),  # 32 x 42 patches pooled 2x2
    "clip_vitb16": (768, 14, 18),
    "siglip_vitb16": (768, 14, 18),
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_feature_map_shape(name):
    enc = build_encoder(name, pretrained=False)
    with torch.no_grad():
        out = enc(torch.randn(2, 3, 480, 640))["feature_map"]
    assert out.shape == (2, *EXPECTED[name])
    assert enc.out_channels == EXPECTED[name][0]


def test_frozen_encoder_has_no_trainable_params_and_stays_in_eval():
    enc = build_encoder("dinov2_vits14", freeze=True, pretrained=False)
    assert not any(p.requires_grad for p in enc.parameters())
    enc.train()
    assert not enc.encoder.training
    x = torch.randn(1, 3, 480, 640, requires_grad=True)
    out = enc(x)["feature_map"]
    assert not out.requires_grad  # no graph through the frozen encoder


def test_finetuned_encoder_gets_gradients():
    enc = build_encoder("resnet18_scratch", pretrained=False)
    enc(torch.randn(1, 3, 480, 640))["feature_map"].mean().backward()
    assert all(p.grad is not None for p in enc.parameters() if p.requires_grad)


def test_frozen_scratch_is_rejected():
    with pytest.raises(ValueError):
        build_encoder("resnet18_scratch", freeze=True, pretrained=False)


def test_clip_renormalization():
    """ImageNet-normalized input must become CLIP-normalized input."""
    enc = build_encoder("clip_vitb16", pretrained=False)
    raw = torch.rand(1, 3, 4, 4)
    imagenet = (raw - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    cfg = enc.vit.pretrained_cfg
    clip = (raw - torch.tensor(cfg["mean"]).view(1, 3, 1, 1)) / torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    assert torch.allclose(imagenet * enc.scale + enc.shift, clip, atol=1e-5)
    assert cfg["mean"] != IMAGENET_MEAN  # otherwise this test proves nothing


def test_unknown_encoder():
    with pytest.raises(KeyError):
        build_encoder("vgg16")
