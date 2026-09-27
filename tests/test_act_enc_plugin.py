"""The act_enc LeRobot plugin. Needs lerobot 0.6.1 (Python >= 3.12) and the
plugin installed:  pip install -e plugins/lerobot_policy_act_enc
Encoder weights are not downloaded (random init); only wiring is tested."""

import functools

import pytest

pytest.importorskip("lerobot")
plugin = pytest.importorskip("lerobot_policy_act_enc")
torch = pytest.importorskip("torch")

from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402

import avr.encoders  # noqa: E402
import lerobot_policy_act_enc.modeling_act_enc as modeling  # noqa: E402
from lerobot_policy_act_enc import ActEncConfig  # noqa: E402

B, CHUNK = 2, 100


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch):
    monkeypatch.setattr(modeling, "build_encoder", functools.partial(avr.encoders.build_encoder, pretrained=False))


def make_config(encoder, freeze=False):
    return ActEncConfig(
        encoder=encoder,
        freeze_encoder=freeze,
        pretrained_backbone_weights=None,  # the replaced ResNet is never used
        device="cpu",
        input_features={
            "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
    )


def train_batch():
    return {
        "observation.images.top": torch.randn(B, 3, 480, 640),
        "observation.state": torch.randn(B, 14),
        "action": torch.randn(B, CHUNK, 14),
        "action_is_pad": torch.zeros(B, CHUNK, dtype=torch.bool),
    }


@pytest.mark.parametrize("encoder,freeze", [("dinov2_vits14", True), ("dinov2_vits14", False), ("resnet18_scratch", False)])
def test_train_step_gradients(encoder, freeze):
    policy = modeling.ActEncPolicy(make_config(encoder, freeze))
    policy.train()
    loss, _ = policy.forward(train_batch())
    loss.backward()
    backbone = [p for n, p in policy.named_parameters() if n.startswith("model.backbone")]
    assert backbone, "encoder must live under model.backbone"
    if freeze:
        assert all(not p.requires_grad for p in backbone)
    else:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backbone)
    assert policy.model.encoder_img_feat_input_proj.weight.grad is not None
    groups = policy.get_optim_params()
    assert len(groups) == (1 if freeze else 2)


def test_select_action_and_roundtrip(tmp_path):
    policy = modeling.ActEncPolicy(make_config("dinov2_vits14", freeze=True))
    policy.eval()
    obs = {k: v[:1] for k, v in train_batch().items() if k.startswith("observation")}
    with torch.no_grad():
        action = policy.select_action(obs)
    assert action.shape == (1, 14)

    policy.save_pretrained(tmp_path)
    loaded = modeling.ActEncPolicy.from_pretrained(tmp_path)
    loaded.eval()
    assert loaded.config.encoder == "dinov2_vits14" and loaded.config.freeze_encoder
    policy.reset()
    loaded.reset()
    with torch.no_grad():
        assert torch.allclose(policy.select_action(obs), loaded.select_action(obs), atol=1e-5)


def test_lerobot_factory_resolves_plugin():
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    assert get_policy_class("act_enc") is modeling.ActEncPolicy
    stats = {
        "observation.state": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "action": {"mean": torch.zeros(14), "std": torch.ones(14)},
        "observation.images.top": {"mean": torch.zeros(3, 1, 1), "std": torch.ones(3, 1, 1)},
    }
    pre, post = make_pre_post_processors(make_config("clip_vitb16", freeze=True), dataset_stats=stats)
    assert pre is not None and post is not None


def test_invalid_configs():
    with pytest.raises(ValueError):
        make_config("vgg16")
    with pytest.raises(ValueError):
        make_config("resnet18_scratch", freeze=True)
