"""LeRobot policy plugin: ACT with a swappable visual encoder (policy type "act_enc")."""

from lerobot_policy_act_enc.configuration_act_enc import ActEncConfig

try:  # image transform "ShadowEdit" (run B); LeRobot imports this plugin before parsing the config
    from avr.shadow_aug import register as _register_shadow_edit

    _register_shadow_edit()
except ImportError:  # the avr package is not installed: the policy still works, only the transform is missing
    pass

__all__ = ["ActEncConfig"]
