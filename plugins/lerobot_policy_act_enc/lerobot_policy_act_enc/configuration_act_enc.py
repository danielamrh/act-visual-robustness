from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.policies.act.configuration_act import ACTConfig

from avr.encoders import ENCODERS


@PreTrainedConfig.register_subclass("act_enc")
@dataclass
class ActEncConfig(ACTConfig):
    """ACT with the image backbone replaced by `avr.encoders.build_encoder(encoder)`.

    Everything else (transformer, VAE, chunking, optimizer preset with
    `optimizer_lr_backbone` for the encoder) is inherited from ACT unchanged.
    """

    encoder: str = "resnet18_imagenet"
    freeze_encoder: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {self.encoder!r}; choose from {sorted(ENCODERS)}")
        if self.freeze_encoder and self.encoder == "resnet18_scratch":
            raise ValueError("freezing a randomly initialized encoder makes no sense")
