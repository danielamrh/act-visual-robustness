"""Swappable visual encoders for ACT.

ACT turns every cell of the backbone's final feature map into one transformer
token (1x1 conv to `dim_model` + 2D sinusoidal position embedding). Each
encoder here therefore returns `{"feature_map": (B, C, h, w)}` like LeRobot's
ResNet backbone, so only the image encoder changes and the rest of ACT stays
identical.

Inputs arrive normalized with ImageNet mean/std (LeRobot's `use_imagenet_stats`);
encoders pretrained with other statistics (CLIP, SigLIP) re-normalize.

| name               | backbone                        | pretraining              | tokens at 480x640 input |
|--------------------|---------------------------------|--------------------------|-------------------------|
| resnet18_imagenet  | ResNet18 (FrozenBatchNorm)      | ImageNet-1k supervised   | 15 x 20 = 300           |
| resnet18_scratch   | ResNet18 (GroupNorm)            | none                     | 15 x 20 = 300           |
| dinov2_vits14      | ViT-S/14                        | DINOv2 self-supervised   | 16 x 21 = 336 (224x294) |
| clip_vitb16        | ViT-B/16                        | CLIP image-text          | 14 x 18 = 252 (224x288) |
| siglip_vitb16      | ViT-B/16                        | SigLIP image-text        | 14 x 18 = 252 (224x288) |

ViTs see a resized image (their pretraining resolution, aspect ratio kept),
which also keeps the token count close to the ResNet's 300.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EncoderSpec:
    kind: str  # "resnet" | "timm"
    timm_name: str | None = None
    image_size: tuple[int, int] | None = None  # (H, W) fed to the encoder; None = native


ENCODERS = {
    "resnet18_imagenet": EncoderSpec("resnet"),
    "resnet18_scratch": EncoderSpec("resnet"),
    "dinov2_vits14": EncoderSpec("timm", "vit_small_patch14_dinov2.lvd142m", (224, 294)),
    "clip_vitb16": EncoderSpec("timm", "vit_base_patch16_clip_224.openai", (224, 288)),
    "siglip_vitb16": EncoderSpec("timm", "vit_base_patch16_siglip_224.webli", (224, 288)),
}


class ResNetEncoder(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        import torchvision
        from torchvision.models._utils import IntermediateLayerGetter
        from torchvision.ops.misc import FrozenBatchNorm2d

        if pretrained:
            # identical to LeRobot's ACT backbone
            model = torchvision.models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1", norm_layer=FrozenBatchNorm2d)
        else:
            # frozen BatchNorm statistics would be meaningless without pretraining; GroupNorm
            # is the usual choice for training ResNets from scratch in robot learning (e.g. Diffusion Policy)
            model = torchvision.models.resnet18(weights=None, norm_layer=lambda c: nn.GroupNorm(c // 16, c))
        self.body = IntermediateLayerGetter(model, return_layers={"layer4": "feature_map"})
        self.out_channels = model.fc.in_features

    def forward(self, x):
        return self.body(x)


class TimmViTEncoder(nn.Module):
    """Patch tokens of a timm ViT, reshaped to a (B, C, h, w) grid (prefix tokens dropped)."""

    def __init__(self, timm_name: str, image_size: tuple[int, int], pretrained: bool = True):
        super().__init__()
        import timm

        self.vit = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.image_size = tuple(image_size)
        self.out_channels = self.vit.num_features
        self.patch = self.vit.patch_embed.patch_size[0]
        cfg = self.vit.pretrained_cfg
        mean, std = torch.tensor(cfg.get("mean", IMAGENET_MEAN)), torch.tensor(cfg.get("std", IMAGENET_STD))
        # x_model = (x_imagenet * std_in + mean_in - mean) / std  ==  x_imagenet * scale + shift
        scale = torch.tensor(IMAGENET_STD) / std
        shift = (torch.tensor(IMAGENET_MEAN) - mean) / std
        self.register_buffer("scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("shift", shift.view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False, antialias=True)
        x = x * self.scale + self.shift
        tokens = self.vit.forward_features(x)[:, self.vit.num_prefix_tokens :]
        h, w = self.image_size[0] // self.patch, self.image_size[1] // self.patch
        return {"feature_map": tokens.transpose(1, 2).reshape(x.shape[0], -1, h, w)}


class FrozenEncoder(nn.Module):
    """Wraps an encoder: no gradients, always in eval mode (dropout / norm statistics fixed)."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder.eval()
        self.out_channels = encoder.out_channels
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        return self.encoder(x)


def build_encoder(name: str, freeze: bool = False, pretrained: bool = True) -> nn.Module:
    """`pretrained=False` skips downloading weights (e.g. when a checkpoint is loaded right after)."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; choose from {sorted(ENCODERS)}")
    spec = ENCODERS[name]
    if spec.kind == "resnet":
        encoder = ResNetEncoder(pretrained=pretrained and name == "resnet18_imagenet")
    else:
        encoder = TimmViTEncoder(spec.timm_name, spec.image_size, pretrained=pretrained)
    if freeze:
        if name == "resnet18_scratch":
            raise ValueError("a frozen randomly initialized encoder makes no sense")
        encoder = FrozenEncoder(encoder)
    return encoder
