from torch import nn

from lerobot.policies.act.modeling_act import ACTPolicy

from avr.encoders import build_encoder
from lerobot_policy_act_enc.configuration_act_enc import ActEncConfig


class ActEncPolicy(ACTPolicy):
    config_class = ActEncConfig
    name = "act_enc"

    def __init__(self, config: ActEncConfig, **kwargs):
        super().__init__(config, **kwargs)
        encoder = build_encoder(config.encoder, freeze=config.freeze_encoder)
        # ACT reads `model.backbone(img)["feature_map"]` and projects it with a 1x1 conv; both are
        # replaced. Parameters under `model.backbone` keep getting `optimizer_lr_backbone`.
        self.model.backbone = encoder
        self.model.encoder_img_feat_input_proj = nn.Conv2d(encoder.out_channels, config.dim_model, kernel_size=1)

    def get_optim_params(self):
        # a frozen encoder leaves the backbone parameter group empty
        return [group for group in super().get_optim_params() if group["params"]]
